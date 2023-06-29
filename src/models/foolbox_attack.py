"""
Implements several foolbox attacks to evaluate the robustness of a given model.
Code adaped from: https://github.com/bethgelab/foolbox/blob/master/examples/single_attack_pytorch_resnet18.py
"""


import numpy as np
import torchvision.models as models
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2PGD, L2CarliniWagnerAttack
from torchvision import transforms
import torch
import logging
from datetime import datetime
from sacred import Experiment
import seml
from models import wideresnet
from data_loader import get_loader
from data_loader_subset import  get_loader as get_loader_TL


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



def foolbox_attack(model_path, num_classes, data_dir, mode, max_nr_img, num_workers,
                   batch_size, save_adversarial_examples, attack_params, epsilons, fc_layers, TL_label_map=None,
                   debug = False, model_path_layers=None, data_set = ''):
    """
    Args:
        - model_path (str or model): Path to saved model parameters or model itself
        - num_classes (int): Number of classes
        - data_dir (str): Path to dataset
        - mode (str): Data subset from which to generate the adversarial examples: 'train', 'val', 'test'
        - max_nr_img (int): Maximum number of adversarial samples to generate
        - num_workers (int): Number of workers
        - batch_size (int): Batch size
        - save_adversarial_examples (bool): If True saves adversarial examples as .npy file
        - attack_params (list): List of attack parameters, either of form ['PGD', pgd_stepsize, list of epsilons]
                         or ['CW', binary_search_steps, steps, stepsize,  confidence, initial_const, abort_early]
    """
    if model_path_layers != None:
        model_path, fc_layers = model_path_layers

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('working on', device)
    #device = 'cpu'

    if type(model_path) == str:
        # path to saved model parameters was passed as input
        model = wideresnet(layers=32, widening_factor=10, num_classes=num_classes,
                           fc_layers=fc_layers).to(device)
        checkpoint = torch.load(model_path)#, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
    else:
        # model was passed as input
        model = model_path

    model.eval()

    # Specify ResNet preprocessing
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    bounds = (0, 1)
    fmodel = PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

    attack_transforms = transforms.ToTensor()

    if TL_label_map != None:
        nat_loader = get_loader_TL(mode, data_dir, transform=attack_transforms, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers, max_nr_img=max_nr_img, label_map=TL_label_map)
    else:
        nat_loader = get_loader(mode, data_dir, transform=attack_transforms, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, max_nr_img=max_nr_img, data_set = data_set)

    # define the attack
    if attack_params[0] == 'PGD':
        pgd_stepsize = attack_params[1]
        attack = LinfPGD(steps=pgd_stepsize)
        print('PGD-Linf', attack)
    elif attack_params[0] == 'PGD-L2':
        pgd_stepsize = attack_params[1]
        attack = L2PGD(steps=pgd_stepsize)
        print('PGD-L2', attack)
    elif attack_params[0] == 'CW':
        binary_search_steps, steps, stepsize, confidence, initial_const, abort_early = attack_params[1:]
        attack = L2CarliniWagnerAttack(binary_search_steps=binary_search_steps,
                                       steps=steps,
                                       stepsize=stepsize,
                                       confidence=confidence,
                                       initial_const=initial_const,
                                       abort_early=abort_early)
        print('CW', attack)
    else:
        print('Please choose either PGD or CW as an attack method.')


    #raw_advs = []
    clipped_advs = []
    success = []
    batch_accs = []

    for images, labels in nat_loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_accs.append(accuracy(fmodel, images, labels))

        batch_raw_advs, batch_clipped_advs, batch_success = attack(fmodel, images, labels, epsilons=epsilons)
        if debug:
            for eps, adv in zip(epsilons, batch_clipped_advs):
                print('\nEpsilons:', eps)
                print('L2_min:', torch.min(torch.norm(adv, dim = (1,2,3), p=2)))
                print('L2_avg:', torch.mean(torch.norm(adv, dim = (1,2,3), p=2)))
                print('L2_max:', torch.max(torch.norm(adv, dim = (1,2,3), p=2)))


                print('Linf_min:', torch.min(torch.norm(adv, dim = (1,2,3), p=np.inf)))
                print('Linf_avg:', torch.mean(torch.norm(adv, dim = (1,2,3), p=np.inf)))
                print('Linf_max:', torch.max(torch.norm(adv, dim = (1,2,3), p=np.inf)))

        #raw_advs.append(np.concatenate(batch_raw_advs, axis = 0))
        clipped_advs.append(torch.cat(batch_clipped_advs, axis = 0))
        success.append(batch_success.type(torch.float32))
        print("robust accuracy for perturbations with")
        for eps, advs_ in zip(epsilons, batch_clipped_advs):
            acc2 = accuracy(fmodel, advs_, labels)
            print(f"  Linf norm ≤ {eps:<6}: {acc2 * 100:4.1f} %")
            if acc2 == 0:
                break

    clean_accuracy = np.mean(batch_accs)
    print(f"clean accuracy:  {clean_accuracy * 100:.1f} %")

    if save_adversarial_examples:
        clipped_advs_save = torch.cat(clipped_advs, axis=0)
        now = datetime.now()
        run_id = str(now)[:16].replace('-', '_').replace(' ', '_').replace(':', '_')  # writer_log_dir.split('/')[-1]
        np.save('src/ARTL/adversarial_examples/clipped_advs_'+str(run_id), clipped_advs_save.cpu().numpy())
    success = torch.cat(success, axis=1)

    adv_eps = {}
    robust_accuracy = 1 - torch.mean(success, axis=1)
    print("robust accuracy for perturbations with")
    for eps, acc in zip(epsilons, robust_accuracy):
        adv_eps[eps] = acc
        if attack_params[0] == 'PGD':
            print(f"  Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        elif attack_params[0] == 'PGD-L2':
            print(f"  L2 norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        elif attack_params[0] == 'CW':
            print(f"  L2 norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")


    results = {
        'clean_accuracy': float(clean_accuracy),
    }
    if attack_params[0] in ['PGD', 'PGD-L2']:
        for key in adv_eps.keys():
            results[str(pgd_stepsize)+'_'+str(key)] = float(adv_eps[key])
    elif attack_params[0] == 'CW':
        for key in adv_eps.keys():
            results[str(key)] = float(adv_eps[key])
    print(results)

    return results


@ex.automain
def run(model_path, num_classes, data_dir, mode, max_nr_img, num_workers,
                   batch_size, save_adversarial_examples, attack_params, epsilons, comment, fc_layers, TL_label_map,
        debug, model_path_layers, data_set = ''):

    logging.info('Received the following configuration:')
    logging.info(f'Data dir: {data_dir}, Model path: {model_path}, num_classes: {num_classes}, '
                 f'batch_size: {batch_size}, attack_params:{attack_params}, epsilons:{epsilons},'
                 f'max_nr_img: {max_nr_img}, num_workers:{num_workers}, mode: {mode}, data_set: {data_set}')

    results = foolbox_attack(model_path, num_classes, data_dir, mode, max_nr_img, num_workers,
               batch_size, save_adversarial_examples, attack_params, epsilons, fc_layers, TL_label_map=TL_label_map,
                             debug = debug,
                             model_path_layers=model_path_layers, data_set=data_set)
    results['comment'] = comment

    return results
          
