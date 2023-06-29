"""
Main training function for training a full neural network.
"""

import os
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import transforms
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2CarliniWagnerAttack, L2PGD
import logging
from sacred import Experiment
import seml
from LCP_utils import AggMo, NpyDataset
from models import project_network_weights, FCNet, wideresnet
from data_loader import get_loader
from utils import write_json, copy_file
from foolbox_attack import foolbox_attack

# SEML setup
ex = Experiment()
seml.setup_logger(ex)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

def accuracy(outputs, targets, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)
    _, preds = outputs.topk(maxk, 1, True, True)
    preds = preds.t()
    correct = preds.eq(targets.view(1, -1).expand_as(preds))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(data_dir, model_path, num_classes, log_step, num_epochs, regularization, batch_size,
          num_workers, learning_rate, patience, early_stopping, adv, start_epoch_adv,
          attack_params, eval_attack, attack_params_eval, epsilons, sigma, max_nr_images, fc_layers,
          WN_initialization, data_set):


    # Image preprocessing, normalization for the pretrained resnet
    normalize = transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))

    # data augmentation
    augm_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ])
    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # data loading
    train_loader = get_loader('train', data_dir, transform=augm_transforms, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, max_nr_img=max_nr_images, data_set=data_set)
    val_loader = get_loader('val', data_dir, transform=eval_transforms, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, max_nr_img=max_nr_images, data_set=data_set)
    test_loader = get_loader('test', data_dir, transform=eval_transforms, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, max_nr_img=max_nr_images, data_set=data_set)

    print('train data size: {}, validation data size: {}, test data size: {}'.format(len(train_loader),
                                                                                     len(val_loader),
                                                                                     len(test_loader)))

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training on', device)

    # Build the models
    model = wideresnet(layers=32, widening_factor = 10, num_classes = 10, fc_layers=fc_layers, device=device).to(device)

    # Loss and optimizer - Full training
    criterion = nn.CrossEntropyLoss() # input expected to contain raw, unnormalized scores for each class
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=regularization
    )

    # decrease learning rate if validation accuracy has not increased
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=1/4, patience=patience, verbose=True,
    )

    now = datetime.now()
    total_step = len(train_loader)
    step = 1
    best_val_acc = 0.0
    best_val_acc_loss = 0.0
    best_train_acc = 0.0
    best_train_acc_loss = 0.0
    prev_val_acc = 0
    early_stop = num_epochs
    val_accs = []
    models = []

    # define mean and std for preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

    # in case of adversarial training: define attack and number of adversarial examples per batch
    if adv:
        if attack_params[0] == 'PGD':
            pgd_stepsize = attack_params[1]
            attack = LinfPGD(steps=pgd_stepsize)
        elif attack_params[0] == 'PGD-L2':
            pgd_stepsize = attack_params[1]
            attack = L2PGD(steps=pgd_stepsize)
            print('PGD - L2', attack)
        elif attack_params[0] == 'CW':
            binary_search_steps, steps, stepsize, confidence, initial_const, abort_early = attack_params[1:]
            attack = L2CarliniWagnerAttack(binary_search_steps=binary_search_steps,
                                           steps=steps,
                                           stepsize=stepsize,
                                           confidence=confidence,
                                           initial_const=initial_const,
                                           abort_early=abort_early)
        else:
            adv = False
            print('Please choose either PGD or CW as an attack method.')

    if fc_layers[0] == 'WN' and WN_initialization:
        model = WN_weight_initialization(train_loader, model, fc_layers, sigma, device)

    for epoch in range(num_epochs):
        for i, (images, targets) in enumerate(train_loader):

            # Set mini-batch dataset
            images = images.to(device)
            targets = targets.to(device)


            if adv and epoch > start_epoch_adv:
                # choose indices
                nr_adv_samples = int(4*np.ceil(len(targets) / 10))
                adv_idx = np.random.randint(0, len(targets)-1, nr_adv_samples)

                # generate adversarial examples
                model.eval()
                fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
                _, clipped_advs, _ = attack(fmodel,
                                            images[adv_idx],
                                            targets[adv_idx],
                                            epsilons=epsilons)

                images[adv_idx] = torch.cat(clipped_advs, axis = 0)

                gaussian_noise_idx = [i for i in range(len(targets)) if i not in adv_idx]

            else:
                gaussian_noise_idx = range(len(targets))


            # manual preprocessing
            images = images.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

            if sigma:
                # add Gaussian noise
                noise = torch.randn_like(images[gaussian_noise_idx], device=device) * sigma
                images[gaussian_noise_idx] += noise

            images.requires_grad = True

            model.train()
            # Forward, backward and optimize
            outputs, feats = model(images, feature = True)
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()

            if fc_layers[0] =='LC':
                # On forward
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e8)

            optimizer.step()

            if fc_layers[0] =='LC':
                # On update
                project_network_weights(model, {"turned_on": True,
                                                "type": "l_inf_projected",
                                                "bjorck_beta": 0.5,
                                                "bjorck_iter": 1,
                                                "bjorck_order": 1}, device)

            if fc_layers[0] == 'WN':
                weight_norm_threshold = fc_layers[2]
                layer_sizes = fc_layers[1]['layer_sizes']
                with torch.no_grad():
                    for l in range(len(layer_sizes)):
                        torch.clamp(model.fc[2 * l].weight_g.data, max=weight_norm_threshold,
                                    out=model.fc[2 * l].weight_g.data)
            outputs = outputs.float()
            loss = loss.float()

            batch_accuracy = accuracy(outputs.data, targets)[0]

            iteration_time = datetime.now() - now

            step += 1

            if (i + 1) % log_step == 0:
                if float(batch_accuracy) > best_train_acc:
                    best_train_acc = float(batch_accuracy)
                    best_train_acc_loss = float(loss)
                print_training_info(num_epochs, batch_accuracy, epoch, i, iteration_time, loss,
                                    step, total_step)
            now = datetime.now()

        # validation step after full epoch
        val_acc, val_loss = model_eval('Validation', criterion, device, model, val_loader, fc_layers,
                                        step)
        lr_scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_loss = val_loss

        if epoch >= early_stopping and val_acc < min(val_accs[epoch - early_stopping:-1]):
            prev_val_accs = val_accs[epoch - early_stopping:-1]
            early_stop = epoch
            print('Early stopping after {} epochs.'.format(epoch))
            model = models[np.argmax(prev_val_accs)]
            break

        val_accs.append(val_acc)
        models.append(model)
        if len(models) > early_stopping:
            models.remove(models[0])

    test_acc, test_loss = model_eval('Test', criterion, device, model, test_loader, fc_layers, step)

    if adv:
        naming_list = [attack_params[0], attack_params[1], epsilons[0], learning_rate, regularization, num_classes]
    else:
        naming_list = [num_epochs, learning_rate, regularization, num_classes]

    results = {}
    if eval_attack:
        for attack_params in attack_params_eval:
            epsilons_eval = attack_params[-1]
            res = foolbox_attack(model, num_classes, data_dir, 'test', 10000, num_workers,
                                 batch_size, False, attack_params[:-1], epsilons_eval, fc_layers, data_set = data_set)
            for key in res.keys():
                results[key] = res[key]

    run_id = save_model_checkpoint(model_path, epoch, model, best_val_acc, naming_list)
    return best_train_acc, best_train_acc_loss, best_val_acc, best_val_acc_loss, test_acc, test_loss, run_id, early_stop, results

def WN_weight_initialization(train_loader, model, fc_layers, sigma, device):
    model.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    # Re-initialize weight_v
    layer_sizes = fc_layers[1]['layer_sizes']
    with torch.no_grad():
        for l in range(len(layer_sizes)):
            torch.nn.init.normal_(model.fc[2 * l].weight_v.data, mean=0.0, std=0.05)

        for i, (images, targets) in enumerate(train_loader):
            if i == 0:
                # Set mini-batch dataset
                images = images.to(device)
                targets = targets.to(device)

                images = images.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

                if sigma:
                    gaussian_noise_idx = range(len(targets))
                    noise = torch.randn_like(images[gaussian_noise_idx], device=device) * 0.1
                    images[gaussian_noise_idx] += noise

                _, y = model(images, feature = True)
                print(y.shape)
                y=y.T
                for l in range(len(layer_sizes)):

                    # Perform custom forward pass
                    print(model.fc[2*l].weight_v.data.shape, y.shape, torch.norm(model.fc[2*l].weight_v.data, dim=1).T.shape)
                    t = torch.mm(model.fc[2*l].weight_v.data, y)
                    print(t.shape, torch.norm(model.fc[2*l].weight_v.data, dim = 1).reshape(-1,1).shape)
                    t = torch.div(t, torch.norm(model.fc[2*l].weight_v.data, dim = 1).reshape(-1,1))
                    t_mean = torch.mean(t, dim = 1).reshape(-1,1)
                    print('t_mean:', t_mean.shape)
                    t_std = torch.std(t, dim = 1).reshape(-1,1)
                    print('t_std:', t_std.shape)

                    if len(layer_sizes) > l+1:
                        preact = (t-t_mean)/t_std
                        y = model.fc[2*l+1](preact.T).T

                    # Initialize weights
                    model.fc[2*l].weight_g.data = 1/t_std
                    model.fc[2*l].bias.data = -t_mean.flatten()/t_std.flatten()

                    print(f'layer {l} - weight_g initialization:', model.fc[2 * l].weight_g.data)
                    print(f'layer {l} - bias initialization:', model.fc[2 * l].bias.data)

    return model

def save_model_checkpoint(model_path, epoch, model, best_val_acc, naming_list):
    now = datetime.now()
    run_id = str(now)[:16].replace('-','_').replace(' ','_').replace(':','_')
    model_dir = os.path.join(model_path, run_id)
    i = 1
    while os.path.isdir(model_dir):
        run_id = run_id.split('_')[0] + '_' + str(i)
        model_dir = os.path.join(model_path, run_id)
        i += 1

    os.makedirs(model_dir, exist_ok=True)
    if len(naming_list) == 6:
        model_path = os.path.join(model_dir,
                                  f'{naming_list[0]}_{naming_list[1]}_{naming_list[2]}_{naming_list[3]}_{naming_list[4]}_{naming_list[5]}.pt')
    else:
        model_path = os.path.join(model_dir,
                                  f'{naming_list[0]}_{naming_list[1]}_{naming_list[2]}_{naming_list[3]}.pt')
    torch.save(model.state_dict(), model_path)

    model_info = {
        'epoch': epoch,
        'best_val_acc': best_val_acc,
        'model_str': str(model)
    }
    json_path = os.path.join(model_dir, 'info.json')
    write_json(model_info, json_path)

    src_model_file = os.path.join(os.path.dirname(__file__), 'models.py')
    dest_model_file = os.path.join(model_dir, 'models.py')
    copy_file(src_model_file, dest_model_file)

    print(f'New checkpoint saved at {model_path}')
    return run_id


def print_training_info(num_epochs, batch_accuracy, epoch, i, iteration_time, loss,
                        step, total_step):#, writer):
    log_info = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}, Iteration time: {}'.format(
        epoch, num_epochs, i + 1, total_step, loss.item(), batch_accuracy, iteration_time
    )
    print(log_info)


def model_eval(mode, criterion, device, model, data_loader, fc_layers, step):

    now = datetime.now()

    # define mean and std for preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    model.eval()

    with torch.no_grad():
        loss_values = []
        acc_values = []

        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)

            # manual preprocessing
            images = images.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])

            outputs = model(images)
            loss = criterion(outputs, targets)
            loss_values.append(loss.item())

            if fc_layers[0] == 'LC':
                # On forward
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e8)

            acc_values.append(accuracy(outputs.data, targets)[0])

        val_loss = sum(loss_values) / len(loss_values)
        val_accuracy = sum(acc_values)/len(acc_values)
        validation_time = datetime.now() - now

        print('{} - Loss: {:.3f}, Acc: {:.3f}, Time: {}'
                .format(mode, val_loss, val_accuracy, validation_time))
    return float(val_accuracy), val_loss



@ex.automain
def run(data_dir, model_path, num_classes, log_step, num_epochs, regularization, batch_size, num_workers, learning_rate,
        patience, early_stopping, adv, start_epoch_adv, attack_params, eval_attack, attack_params_eval, epsilons,
        sigma, max_nr_images, fc_layers, data_set=False, WN_initialization=False):

    logging.info('Received the following configuration:')
    logging.info(f'Data dir: {data_dir}, Model path: {model_path}, num_epochs: {num_epochs}, sigma: {sigma},'
                 f'regularization: {regularization}, early_stopping: {early_stopping}, epsilons: {epsilons},'
                 f'batch_size: {batch_size}, num_workers: {num_workers}, learning_rate: {learning_rate},'
                 f'patience: {patience}, fc_layers: {fc_layers}, adv: {adv}, data_set: {data_set},'
                 f'max_nr_images: {max_nr_images}, WN_initialization: {WN_initialization}')

    best_train_acc, best_train_loss, best_val_acc,\
    best_val_acc_loss,test_acc, test_loss, \
    run_id, early_stop, res = train(data_dir, model_path, num_classes, log_step, num_epochs, regularization,
                                    batch_size, num_workers, learning_rate, patience, early_stopping, adv,
                                    start_epoch_adv, attack_params, eval_attack, attack_params_eval, epsilons,
                                    sigma, max_nr_images, fc_layers, WN_initialization, data_set)

    results = {
        'best_train_acc': best_train_acc,
        'best_train_loss': best_train_loss,
        'best_val_acc': best_val_acc,
        'best_val_acc_loss': best_val_acc_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'early_stop': early_stop,
        'run_id': run_id
    }

    for key in res.keys():
        results[key] = res[key]

    return results
