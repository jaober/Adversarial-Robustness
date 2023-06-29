"""
Implements retraining of the last k blocks of a pre-trained WideResNet model.
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
from foolbox.attacks import LinfPGD
import logging
from sacred import Experiment
import seml
from LCP_utils import AggMo, NpyDataset
from models import project_network_weights, FCNet, wideresnet
from data_loader import get_loader
from data_loader_subset import  get_loader as get_loader_TL
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


def accuracy_train(outputs, targets, topk=(1,)):
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


def train(data_dir, model_path_layers_unfreeze, num_classes, log_step, num_epochs, regularization, batch_size,
          num_workers, learning_rate, patience, early_stopping, hinge,
          freeze_epochs, attack_params_list, sigma, max_nr_images, TL_label_map, LC_margin, WN_initialization):

    model_path, fc_layers, nr_unfreeze, save_model = model_path_layers_unfreeze
    print('model_path, fc_layers, nr_unfreeze, save_model', model_path, fc_layers, nr_unfreeze, save_model)

    # data augmentation
    augm_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ])
    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])


    if TL_label_map != None:
        # data loading
        train_loader = get_loader_TL('train', data_dir, transform=augm_transforms, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, max_nr_img=max_nr_images, label_map=TL_label_map)
        val_loader = get_loader_TL('val', data_dir, transform=eval_transforms, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, max_nr_img=max_nr_images, label_map=TL_label_map)
        test_loader = get_loader_TL('test', data_dir, transform=eval_transforms, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, max_nr_img=max_nr_images, label_map=TL_label_map)

    else:
        # data loading
        train_loader = get_loader('train', data_dir, transform=augm_transforms, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, max_nr_img=max_nr_images)
        val_loader = get_loader('val', data_dir, transform=eval_transforms, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, max_nr_img=max_nr_images)
        test_loader = get_loader('test', data_dir, transform=eval_transforms, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, max_nr_img=max_nr_images)

    print('train data size: {}, validation data size: {}, test data size: {}'.format(len(train_loader),
                                                                                     len(val_loader),
                                                                                     len(test_loader)))

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('training on', device)

    # Build the models
    model = wideresnet(layers=32, widening_factor = 10, num_classes = 10, fc_layers=fc_layers).to(device)
    if model_path:
        checkpoint = torch.load(model_path)

        if fc_layers[0] == 'WN' and 'fc.bias' in checkpoint.keys():
            checkpoint['fc.0.bias'] = checkpoint.pop('fc.bias')
            checkpoint['fc.0.weight_g'] = checkpoint.pop('fc.weight_g')
            checkpoint['fc.0.weight_v'] = checkpoint.pop('fc.weight_v')

        model.load_state_dict(checkpoint)

    model = reset_layers(model, nr_unfreeze, fc_layers)
    model = freeze_layers(model, nr_unfreeze, fc_layers)


    # Loss and optimizer
    if fc_layers[0] == 'LC':
        margin = LC_margin
        l_constant = fc_layers[1]['l_constant']
        criterion = nn.MultiMarginLoss(margin=margin * l_constant)
        optimizer = AggMo(model.parameters(), lr=learning_rate, momentum=[0.0, 0.9, 0.99], weight_decay=regularization)
    elif hinge:
        margin = LC_margin
        if fc_layers[0] == 'LC':
            k_constant = fc_layers[1]['l_constant']
        elif fc_layers[0] == 'WN':
            k_constant = fc_layers[2]
        criterion = nn.MultiMarginLoss(margin=margin * k_constant)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)

    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)

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
    run_id = ''
    early_stopper = 0
    prev_val_acc = 0
    early_stop = num_epochs
    val_accs = []
    models = []

    # define mean and std for preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)

    if fc_layers[0] == 'WN' and WN_initialization:
        model = WN_weight_initialization(train_loader, model, fc_layers, sigma, device)

    for epoch in range(num_epochs):

        for i, (images, targets) in enumerate(train_loader):

            # Set mini-batch dataset
            images = images.to(device)
            targets = targets.to(device)

            # add Gaussian noise
            if sigma:
                noise = torch.randn_like(images, device=device) * sigma
                images += noise

            # manual preprocessing
            images = images.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
            images.requires_grad = True

            model.train()

            # Forward, backward and optimize
            outputs = model(images)
            loss = criterion(outputs, targets)
            model.zero_grad()
            loss.backward()

            if fc_layers[0] == 'LC':
                # On forward
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e8)

            optimizer.step()

            if fc_layers[0] == 'LC':
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
                        print('Weight_g - Pre-update:', model.fc[2 * l].weight_g.flatten())
                        torch.clamp(model.fc[2 * l].weight_g, max=weight_norm_threshold,
                                    out=model.fc[2 * l].weight_g)
                        print('Weight_g - Post-update:', model.fc[2 * l].weight_g.flatten())

            outputs = outputs.float()
            loss = loss.float()

            batch_accuracy = accuracy_train(outputs.data, targets)[0]

            iteration_time = datetime.now() - now
            step += 1

            if (i + 1) % log_step == 0:
                if float(batch_accuracy) > best_train_acc:
                    best_train_acc = float(batch_accuracy)
                    best_train_acc_loss = float(loss)
                print_training_info(num_epochs, batch_accuracy, epoch, i, iteration_time, loss,
                                    step, total_step)#, writer)
            now = datetime.now()


        # validation step after full epoch
        val_acc, val_loss = model_eval('Validation', criterion, device, model, val_loader, fc_layers,
                                        step)
        lr_scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_acc_loss = val_loss

        if epoch >= early_stopping and val_acc > max(val_accs[epoch-early_stopping:-1]):
            prev_val_accs = val_accs[epoch-early_stopping:-1]
            early_stop = epoch
            print('Early stopping after {} epochs.'.format(epoch))
            model = models[np.argmax(prev_val_accs)]
            break

        val_accs.append(val_acc)
        models.append(model)
        if len(models) > early_stopping:
            models.remove(models[0])

    test_acc, test_loss = model_eval('Test', criterion, device, model, test_loader, fc_layers,
                                       step)
    if save_model:
        naming_list = [nr_unfreeze, freeze_epochs, learning_rate, regularization, num_classes]
        run_id = save_model_checkpoint('models/', epoch, model, best_val_acc, naming_list)
        print('Model saved at:', run_id)

    results = {}
    for attack_params in attack_params_list:
        epsilons_eval = attack_params[-1]
        res = foolbox_attack(model, num_classes, data_dir, 'test', 10000, num_workers,
                             batch_size, False, attack_params[:-1], epsilons_eval, fc_layers, TL_label_map = TL_label_map)
        for key in res.keys():
            results[key] = res[key]

    return best_train_acc, best_train_acc_loss, best_val_acc, best_val_acc_loss, test_acc, test_loss, run_id, early_stop, \
           results, val_accs

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
                    y = nn.functional.relu((t-t_mean)/t_std)

                    # Initialize weights
                    model.fc[2*l].weight_g.data = 1/t_std
                    model.fc[2*l].bias.data = -t_mean.flatten()/t_std.flatten()

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

    if len(naming_list) == 5:
        model_path = os.path.join(model_dir,
                                  f'Exp1_{naming_list[0]}_{naming_list[1]}_{naming_list[2]}_{naming_list[3]}_{naming_list[4]}.pt')
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
                        step, total_step):
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

            acc_values.append(accuracy_train(outputs.data, targets)[0])

        val_loss = sum(loss_values) / len(loss_values)
        val_accuracy = sum(acc_values)/len(acc_values)
        validation_time = datetime.now() - now

        print('{} - Loss: {:.3f}, Acc: {:.3f}, Time: {}'
                .format(mode, val_loss, val_accuracy, validation_time))

    return float(val_accuracy), val_loss


def freeze_layers(model, nr_unfreeze, fc_layers):
    if fc_layers[0] in ['LC', 'WN']:
        nr_fc_layers = len(fc_layers[1]['layer_sizes'])
    else:
        nr_fc_layers = 1
    total_block_nr = 16 + nr_fc_layers
    assert nr_unfreeze <= total_block_nr, 'Error: Block_nr has to be <= 16 + {}!'.format(nr_fc_layers)

    i = 0
    layer_list = [model.layer3, model.layer2, model.layer1]
    l = 2
    nr_freeze = total_block_nr - nr_unfreeze - 1
    fc_nr = 0

    for k in range(nr_freeze):
        if k > total_block_nr - 2 - nr_fc_layers:
            if fc_layers[0] == 'LC':
                layer = model.fc[fc_nr * 3]
            elif fc_layers[0] == 'WN':
                layer = model.fc[fc_nr * 2]

            else:
                layer = model.fc
            layer.requires_grad_(False)
            print('Block', k, ':', 'fc-layer:', fc_nr)
            fc_nr += 1
        elif k == total_block_nr - 2 - nr_fc_layers:
            layer = model.bn3
            layer.requires_grad_(False)
            print('Block', k, ':', 'bn3')
        else:
            for layer in [layer_list[l][i].conv1,
                          layer_list[l][i].conv2]:
                layer.requires_grad_(False)

            for layer in [layer_list[l][i].bn1, layer_list[l][i].bn2]:
                # layer.eval()
                layer.requires_grad_(False)
                layer.requires_grad_(False)
            if i == 4:
                i = 0
                l -= 1
            else:
                i += 1
        k += 1

    if nr_unfreeze < total_block_nr:
        model.conv1.requires_grad_(False)

    return model
  

def unfreeze_layer(model, block_nr):
    assert block_nr <= 18, 'Error: Block_nr has to be <= 18!'
    block_dict = {3: [model.layer3, 4],
                  4: [model.layer3, 3],
                  5: [model.layer3, 2],
                  6: [model.layer3, 1],
                  7: [model.layer3, 0],
                  8: [model.layer2, 4],
                  9: [model.layer2, 3],
                  10: [model.layer2, 2],
                  11: [model.layer2, 1],
                  12: [model.layer2, 0],
                  13: [model.layer1, 4],
                  14: [model.layer1, 3],
                  15: [model.layer1, 2],
                  16: [model.layer1, 1],
                  17: [model.layer1, 0]}
    if block_nr == 2:
        layer = model.bn3
        layer.train()
        print('Block', block_nr, ':', 'bn3')
    elif block_nr == 18:
        model.conv1.requires_grad_(True)
    else:
        for layer in [block_dict[block_nr][0][block_dict[block_nr][1]].conv1,
                      block_dict[block_nr][0][block_dict[block_nr][1]].conv2]:
            layer.requires_grad_(True)
        for layer in [block_dict[block_nr][0][block_dict[block_nr][1]].bn1,
                      block_dict[block_nr][0][block_dict[block_nr][1]].bn2]:
            layer.train()

    return model



def reset_layers(model, nr_unfreeze, fc_layers):
    if fc_layers[0] in ['LC', 'WN']:
        nr_fc_layers = len(fc_layers[1]['layer_sizes'])
    else:
        nr_fc_layers = 1
    assert nr_unfreeze <= 16 + nr_fc_layers, 'Error: Block_nr has to be <= 16 + {}!'.format(nr_fc_layers)

    fc_nr = nr_fc_layers - 1
    k = 1
    layer_nr = 4
    w_res_layer_list = [model.layer1, model.layer2, model.layer3]
    w_res_block_nr = 2
    while k <= nr_unfreeze:
        if k <= nr_fc_layers:
            if fc_layers[0] == 'LC':
                layer = model.fc[fc_nr*3]
            elif fc_layers[0] == 'WN':
                layer = model.fc[fc_nr*2]
            else:
                layer = model.fc
            layer.reset_parameters()
            print('Block', k, ':', 'fc-layer:', fc_nr)
            fc_nr -= 1
        elif k == nr_fc_layers+1:
            layer = model.bn3
            layer.reset_parameters()
            print('Block', k, ':', 'bn3')

        else:
            for layer in [w_res_layer_list[w_res_block_nr][layer_nr].conv1, w_res_layer_list[w_res_block_nr][layer_nr].bn1,
                          w_res_layer_list[w_res_block_nr][layer_nr].conv2, w_res_layer_list[w_res_block_nr][layer_nr].bn2]:
                layer.reset_parameters()
            print('Block ', k, ':', 'layer', w_res_block_nr + 1, 'Rep', layer_nr)
            if layer_nr == 0:
                layer_nr = 4
                w_res_block_nr -= 1
            else:
                layer_nr -= 1
        k += 1
    return model


@ex.automain
def run(data_dir, model_path_layers_unfreeze, num_classes, log_step, num_epochs, regularization, batch_size, num_workers, learning_rate,
        patience, early_stopping, hinge, freeze_epochs, attack_params_list, sigma, max_nr_images, TL_label_map, LC_margin,
        WN_initialization=False):

    logging.info('Received the following configuration:')
    logging.info(f'Data dir: {data_dir}, Model path: {model_path_layers_unfreeze}, num_epochs: {num_epochs}, '
                 f'regularization: {regularization}, early_stopping: {early_stopping}, sigma: {sigma},'
                 f'batch_size: {batch_size}, num_workers:{num_workers}, learning_rate: {learning_rate}, patience:{patience}'
                 f'WN_initialization: {WN_initialization}')


    best_train_acc, best_train_loss, best_val_acc, best_val_acc_loss,test_acc, test_loss, run_id, early_stop, res, val_accs = train(data_dir,
                                                        model_path_layers_unfreeze, num_classes, log_step, num_epochs, regularization,
                                                        batch_size, num_workers, learning_rate, patience,
                                                        early_stopping, hinge, freeze_epochs, attack_params_list,
                                                        sigma, max_nr_images, TL_label_map, LC_margin, WN_initialization)

    results = {
        'best_train_acc': best_train_acc,
        'best_train_loss': best_train_loss,
        'best_val_acc': best_val_acc,
        'best_val_acc_loss': best_val_acc_loss,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'early_stop': early_stop,
        'run_id': run_id,
        'val_accs': val_accs
    }
    for key in res.keys():
        results[key] = res[key]
    return results

