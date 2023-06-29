"""
Implements full certification pipeline for weight normalization and Lipschitz constrained models via median smoothing
or center smoothing. Results are displayed in Theoretical Robustness Bounds.ipynb.
"""

from median_smoothing import *
import numpy as np
import pandas as pd
import os
from itertools import islice
import math
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
   sys.path.append(module_path+"/ARTL")
from data_loader_TL import  get_loader as get_loader_TL
from pathlib import Path
from models import wideresnet, WideResNet_gurobi
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, L2PGD, L2CarliniWagnerAttack
from center_smoothing import *
from torch.nn.functional import normalize


def run_median_smoothing_single_sigma(n, eps, sigma, alpha, model, img_loader, batch_size, max_nr_img, 
                                      attack_params, nr_outputs, l_constant, device):

    # Device configuration
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('working on', device)
    
    # Adapt confthreshold for mutliple dimensions
    conf_threshold_total = alpha
    conf_thres = 1 - conf_threshold_total
    conf_thres = 1- conf_thres/nr_outputs
    print('Conf_thresh.:', conf_thres)
    
    qu, ql = estimated_qu_ql(eps, n, sigma, conf_thres=conf_thres)
    print('Working with qu: {}, ql: {}'.format(qu, ql))
    
    model = model.to(device)
    model.eval()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    ms_results = {}
    eps_acc = []

    img_nrs = 0
    print('Iterating over {} batches.'.format(max_nr_img//batch_size + int(max_nr_img%batch_size > 0)))
        
    # Specify ResNet preprocessing
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    bounds = (0, 1)
    fmodel = PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

    # define the attack
    if attack_params[0] == 'PGD':
        pgd_stepsize = attack_params[1]
        attack = LinfPGD(steps=pgd_stepsize)
        epsilons = attack_params[2]
        print('PGD-Linf', epsilons)
    elif attack_params[0] == 'PGD-L2':
        pgd_stepsize = attack_params[1]
        attack = L2PGD(steps=pgd_stepsize)
        epsilons = attack_params[2]
        print('PGD-L2', epsilons)
    elif attack_params[0] == 'CW':
        binary_search_steps, steps, stepsize, confidence, initial_const, abort_early = attack_params[1:]
        attack = L2CarliniWagnerAttack(binary_search_steps=binary_search_steps,
                                       steps=steps,
                                       stepsize=stepsize,
                                       confidence=confidence,
                                       initial_const=initial_const,
                                       abort_early=abort_early)
        print('CW', epsilons)
    else:
        print('Please choose either PGD or CW as an attack method.')

    
   # with torch.no_grad():
    for images, labels in img_loader:
        bounds_norms = torch.zeros(len(labels)).to(device)
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            # Get features
            images_pred = images.sub(mean[None, :, None, None]).div(std[None, :, None, None])

            orig_out, predictions = model(images_pred, feature = True)           

            print('Iterating over {} images in batch.'.format(len(images)))
            for img_nr in tqdm(range(len(images))):
                x = images_pred[img_nr]
                ms_results[img_nrs + img_nr] = dict()
                ms_results[img_nrs + img_nr]['pred'] = predictions[img_nr].detach().cpu().numpy()
                ms_results[img_nrs + img_nr]['class'] = int(labels[img_nr])
                # preprocessing included in certify --> run on UN-preprocessed data in [0,1]
                y_med, yl, yu = certify(model, images[img_nr].reshape((1,3,32,32)), qu, ql, sigma, n, eps, alpha, batch_size, device)
                ms_results[img_nrs + img_nr][sigma] = [y_med.detach().cpu().numpy()]
                                                        #yl.detach().cpu().numpy(), 
                                                       #yu.detach().cpu().numpy()]
                ms_results[img_nrs + img_nr]['l2_bounds_norm'] = torch.norm(yl-yu).detach().cpu().numpy()
                ms_results[img_nrs + img_nr]['linf_bounds_norm'] = torch.norm(yl-yu, p = np.inf).detach().cpu().numpy()
                bounds_norms[img_nr]=torch.norm(yl - yu, p = np.inf)
                del yl
                del yu
            if l_constant:
                eps_acc.append(theoretical_adversary_ms(orig_out, labels, l_constant * bounds_norms))

                
        batch_raw_advs, batch_clipped_advs, batch_success = attack(fmodel, images, labels, epsilons=epsilons)   
        with torch.no_grad():
            images_pert = batch_clipped_advs[0] # Works bec. one values for epsilon passed only
            images_pert = images_pert.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
            pert_out, pert_features = model(images_pert, feature = True)            
            pred_orig = torch.argmax(orig_out, dim = 1)
            pred_pert = torch.argmax(pert_out, dim = 1)            
            orig_correct = pred_orig.eq(labels).flatten()
            pert_correct = pred_pert.eq(labels).flatten()

        for img_nr in tqdm(range(len(images))):
            ms_results[img_nrs + img_nr]['z_pert'] = pert_features[img_nr].detach().cpu().numpy()
        img_nrs += len(images)

    return ms_results, None


def run_center_smoothing(eps, sigma, model, img_loader, batch_size, max_nr_img, attack_params, nr_outputs, device):

    # Device configuration
    print('working on', device)
    
    model = model.to(device)
    model.eval()
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    ms_results = {}
    eps_acc = []

    img_nrs = 0
    print('Iterating over {} batches.'.format(max_nr_img//batch_size + int(max_nr_img%batch_size > 0)))
    
        
    # Specify ResNet preprocessing
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    bounds = (0, 1)
    fmodel = PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

    # define the attack
    if attack_params[0] == 'PGD':
        pgd_stepsize = attack_params[1]
        attack = LinfPGD(steps=pgd_stepsize)
        epsilons = attack_params[2]
        print('PGD-Linf', epsilons)
    elif attack_params[0] == 'PGD-L2':
        pgd_stepsize = attack_params[1]
        attack = L2PGD(steps=pgd_stepsize)
        epsilons = attack_params[2]
        print('PGD-L2', epsilons)
    elif attack_params[0] == 'CW':
        binary_search_steps, steps, stepsize, confidence, initial_const, abort_early = attack_params[1:]
        attack = L2CarliniWagnerAttack(binary_search_steps=binary_search_steps,
                                       steps=steps,
                                       stepsize=stepsize,
                                       confidence=confidence,
                                       initial_const=initial_const,
                                       abort_early=abort_early)
        print('CW', epsilons)
    else:
        print('Please choose either PGD or CW as an attack method.')

    
    for images, labels in img_loader:
        bounds_norms = torch.zeros(len(labels)).to(device)
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            # Get features
            images_pred = images.sub(mean[None, :, None, None]).div(std[None, :, None, None])

            #labels = labels.to(device)
            orig_out, predictions = model(images_pred, feature = True)           

            print('Iterating over {} images in batch.'.format(len(images)))
            for img_nr in tqdm(range(len(images))):
                x = images_pred[img_nr]
                ms_results[img_nrs + img_nr] = dict()
                ms_results[img_nrs + img_nr]['pred'] = predictions[img_nr].detach().cpu().numpy()
                ms_results[img_nrs + img_nr]['class'] = int(labels[img_nr])
                eps2, z = center_certify(model, x=images[img_nr], sigma=sigma, 
                                      m=10**4, 
                                      n=10**4, 
                                      eps1=eps, delta=0.05, 
                               alpha1=0.01, alpha2=0.01, beta=2, num_steps=10, batch_size=batch_size, device=device)
                ms_results[img_nrs + img_nr][sigma] = [z.detach().cpu().numpy()]
                ms_results[img_nrs + img_nr]['bound'] = eps2.detach().cpu().numpy()
              
        batch_raw_advs, batch_clipped_advs, batch_success = attack(fmodel, images, labels, epsilons=epsilons)   
        with torch.no_grad():
            images_pert = batch_clipped_advs[0] 
            images_pert = images_pert.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
            pert_out, pert_features = model(images_pert, feature = True)            
            pred_orig = torch.argmax(orig_out, dim = 1)
            pred_pert = torch.argmax(pert_out, dim = 1)            
            orig_correct = pred_orig.eq(labels).flatten()
            pert_correct = pred_pert.eq(labels).flatten()

        for img_nr in tqdm(range(len(images))):
            ms_results[img_nrs + img_nr]['z_pert'] = pert_features[img_nr].detach().cpu().numpy()
        img_nrs += len(images)

    return ms_results


def run_MS_for_TL_models(smoothing_method, model_info, loaders, n, eps, sigma, alpha, batch_size, max_nr_img, attack_params, nr_outputs, device):
    ms_results_dict = {}
    theoretical_accuracies = {}
    eps_range = eps_range = np.array([0.001, 0.01, 0.0313, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]) 
                                                       
    for model_name in model_info:
        fc_layers = model_info[model_name]['fc_layers']
        if fc_layers[0] == 'LC':
            l_constant = fc_layers[1]['l_constant']
        else:
            l_constant = False      
        for model_run in [mv for mv in model_info[model_name].keys() if mv != 'fc_layers']:

            data_loader = loaders[model_run]
            if Path(model_info[model_name][model_run]).exists(): # if model file is currently stored on server
                # Get Base Model
                checkpoint = torch.load(model_info[model_name][model_run]) 
                if fc_layers[0] == 'WN' and 'fc.bias' in checkpoint.keys():
                    checkpoint['fc.0.bias'] = checkpoint.pop('fc.bias')
                    checkpoint['fc.0.weight_g'] = checkpoint.pop('fc.weight_g')
                    checkpoint['fc.0.weight_v'] = checkpoint.pop('fc.weight_v')                
                
                model2 = wideresnet(layers=32, widening_factor = 10, num_classes = 10, fc_layers=fc_layers).to(device)
                model2.load_state_dict(checkpoint)
                model2.eval()
                
                display(get_per_class_accuracies(model2, data_loader, device))
    
                if smoothing_method == 'MS': 
                    ms_results, avg_eps_acc = run_median_smoothing_single_sigma(n, eps, sigma, alpha, 
                                                                                model2, data_loader, batch_size, 
                                                                                max_nr_img, attack_params, 
                                                                                nr_outputs, l_constant, device)
                
                elif smoothing_method == 'CS':
                    avg_eps_acc = None
                    ms_results = run_center_smoothing(eps, sigma, model2, data_loader, batch_size, max_nr_img, attack_params, nr_outputs, device)
                        
                print('Working on model {} - {}'.format(model_name, model_run))

                if fc_layers[0] == 'WN':
                    g = model2.fc[0].weight_g
                    v = model2.fc[0].weight_v
                    norm_v = torch.norm(v, dim = 1).reshape((-1,1))
                    W = g*v/norm_v
                    print('g', g.flatten().detach().cpu().numpy())
                    b = model2.fc[0].bias
                elif fc_layers[0] == 'LC':
                    W = model2.fc[0].weight
                    b = model2.fc[0].bias
                else:
                    W = model2.fc.weight
                    b = model2.fc.bias
                print(W[1,1])

                # Print stats on ||z_upper - z_lower|| and ||z - z'||
                ms_results, tightness_df_bounds, tightness_df_pert = run_MS_TL_evaluations_norms(smoothing_method, ms_results, sigma)
                print('\nStats on tightness of bounds: ||z_upper - z_lower||:')
                display(tightness_df_bounds)
                
                print("\nStats on ||z - z'||:")
                display(tightness_df_pert)
                
                # Print stats on logits
                ms_results, accs_df, logits_stats_df, logits_stats_correct_df = get_ms_accuracy_and_logit_stats(smoothing_method, ms_results, sigma, W, b, device, fc_layers[0]=='WN')
                
                print('\nMS Accuracies:')
                display(accs_df)
                
                print('\nStats on logits:')
                display(logits_stats_df)
                
                print('\nStats on logits - Correct:')
                display(logits_stats_correct_df)
                
                if fc_layers[0] == 'LC':
                    l_constant = fc_layers[1]['l_constant']
                    theoretical_accuracies[model_name + ' - ' + model_run] = get_theoretical_lower_bound(model2, data_loader, l_constant, eps_range, device)
                
            ms_results_dict[model_name + ' - ' + model_run] = ms_results
    return ms_results_dict, theoretical_accuracies, avg_eps_acc

                      
def eval_center_smoothing(cs_results):
    """Prints brief summary of Center Smoothing Dertification.
    Input: 
        cs_results: dictionary containing results of Center Smoothing certification, mapping image number to eps2 value
    Output:
        cs_stats_df: Dataframe containing minimum, mean and maximum of eps2 results
    """
    cs_df = pd.DataFrame(cs_results,index = ['bounds']).T
    
    # For how many images has CS certification succeded?
    cs_df_success = cs_df.loc[cs_df.bounds != None]
    print('CS Certification successful for {}/{} images.'.format(len(cs_df_success), len(cs_df)))

    # Get stats on eps2 for the successful certifications
    print('\nCS Certification Stats (on {} images):'.format(len(cs_df_success)))
    cs_stats = {'eps2': {'max': np.max(cs_df_success.bounds),
                         'mean': np.mean(cs_df_success.bounds),
                         'min': np.min(cs_df_success.bounds),}}
    
    cs_stats_df = pd.DataFrame(cs_stats).T
    return cs_stats_df    

                      
def run_MS_TL_evaluations_norms(smoothing_method, ms_results, sigma):
    
    if smoothing_method == 'MS':
        l2_tightness_bounds = []
        linf_tightness_bounds = []
        l2_tightness_pert = []
        linf_tightness_pert = []
        nr_perturbations_below_threshold_l2 = 0
        nr_perturbations_below_threshold_linf = 0
        nr_successfull_perturbations = 0

        for image_nr in ms_results.keys():
            l2_bounds = ms_results[image_nr]['l2_bounds_norm']
            linf_bounds = ms_results[image_nr]['linf_bounds_norm']

            l2_tightness_bounds.append(l2_bounds)
            linf_tightness_bounds.append(linf_bounds)

            nr_successfull_perturbations += 1
            l2_pred_pert = np.linalg.norm(ms_results[image_nr]['z_pert'] - ms_results[image_nr]['pred'])
            linf_pred_pert = np.linalg.norm(ms_results[image_nr]['z_pert'] - ms_results[image_nr]['pred'], ord = np.inf)
            l2_tightness_pert.append(l2_pred_pert)
            linf_tightness_pert.append(linf_pred_pert)

            if l2_pred_pert <= l2_bounds:
                nr_perturbations_below_threshold_l2 += 1

            if linf_pred_pert <= linf_bounds:
                nr_perturbations_below_threshold_linf += 1
                
        tightness_stats_bounds = {'l2': {'min': float(np.min(l2_tightness_bounds)), 
                          'mean': float(np.mean(l2_tightness_bounds)), 
                          'max': float(np.max(l2_tightness_bounds)), 
                          'std': float(np.std(l2_tightness_bounds))},
                   'linf': {'min': float(np.min(linf_tightness_bounds)), 
                            'mean': float(np.mean(linf_tightness_bounds)), 
                            'max': float(np.max(linf_tightness_bounds)), 
                            'std': float(np.std(linf_tightness_bounds))}}
        tightness_df_bounds = pd.DataFrame(tightness_stats_bounds).T
        
        print("Number of ||z - z'||_2 below ||z_upper - z_lower||_2: {}/{}".format(nr_perturbations_below_threshold_l2,
          nr_successfull_perturbations))
        print("Number of ||z - z'||_inf below ||z_upper - z_lower||_inf: {}/{}".format(nr_perturbations_below_threshold_linf,
          nr_successfull_perturbations))
                  
    elif smoothing_method == 'CS':
        tightness_bounds = []
        l2_tightness_pert = []
        linf_tightness_pert = []
        nr_perturbations_below_threshold = 0
        nr_successfull_perturbations = 0

        for image_nr in ms_results.keys():
            bound = ms_results[image_nr]['bound']

            if bound != None:
            
                tightness_bounds.append(bound)

                nr_successfull_perturbations += 1

                l2_pred_pert = np.linalg.norm(ms_results[image_nr]['z_pert'] - ms_results[image_nr]['pred'])
                linf_pred_pert = np.linalg.norm(ms_results[image_nr]['z_pert'] - ms_results[image_nr]['pred'], ord = np.inf)
                l2_tightness_pert.append(l2_pred_pert)
                linf_tightness_pert.append(linf_pred_pert)

                if l2_pred_pert <= bound:
                    nr_perturbations_below_threshold += 1
                
        tightness_stats_bounds = {'eps_2': {'min': float(np.min(tightness_bounds)), 
                  'mean': float(np.mean(tightness_bounds)), 
                  'max': float(np.max(tightness_bounds)), 
                  'std': float(np.std(tightness_bounds))}}
        tightness_df_bounds = pd.DataFrame(tightness_stats_bounds).T
              
        print("Number of ||z - z'||_2 below eps_2-bound: {}/{}".format(nr_perturbations_below_threshold,
              nr_successfull_perturbations))                                                                                          
                                                                                                           
    tightness_stats_pert = {'l2': {'min': float(np.min(l2_tightness_pert)), 
                              'mean': float(np.mean(l2_tightness_pert)), 
                              'max': float(np.max(l2_tightness_pert)), 
                              'std': float(np.std(l2_tightness_pert))},
                       'linf': {'min': float(np.min(linf_tightness_pert)), 
                                'mean': float(np.mean(linf_tightness_pert)), 
                                'max': float(np.max(linf_tightness_pert)), 
                                'std': float(np.std(linf_tightness_pert))}}
    tightness_df_pert = pd.DataFrame(tightness_stats_pert).T
                                                                                                          
    return ms_results, tightness_df_bounds, tightness_df_pert

                                              
def get_per_class_accuracies(model, data_loader, device):
    
    model.eval()
    accs = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6:[], 7: [], 8: [], 9: []}
    
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Get features
            images = images.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
            out = model(images)
            preds = torch.argmax(out, dim = 1)
            correct = preds.eq(labels).flatten()
            
            for i in range(len(correct)):
                accs[int(labels[i])].append(correct[i])    

        full_set_acc = []
        for c in accs.keys():
            full_set_acc.extend(accs[c])
            accs[c] = np.around(float(sum(accs[c])/len(accs[c]))*100, 2)

        accs['All'] = np.around(float(sum(full_set_acc)/len(full_set_acc))*100, 2)

    return accs


def certify(model, x, qu, ql, sigma, n, eps, alpha, batch_size, device):
    """ Certify all coordinates simultaneously for one image. 

    Args:
        - x (torch.Tensor): Single input image of size [channel x width x height]
        - sigma (float): Standard deviation of Gaussian noise
        - n (int): Number of samples for Monte Carlo sampling
        - eps (float): Limit of L2-norm of adversarial perturbation 
        - alpha (float): Confidence level
        - batch_size (int): Batch size
        - device (device): Device
    """
    with torch.no_grad():
        y_hat = add_gaussian_noise_ms(model, x, sigma, n, batch_size, device)
        y_hats = torch.stack(y_hat, axis = 0) #torch.cat(y_hat, axis=1)
        # Sort each coordinate along the batch dimension -> Why? Needed?
        y_hat_sorted = torch.sort(y_hats, dim=0)[0]
        y_median = torch.median(y_hats, axis = 0)[0]
        # Take the qth order statistics
        yu = y_hat_sorted[qu,:]
        yl = y_hat_sorted[ql,:]
    return y_median, yl, yu


def add_gaussian_noise_ms(model2, x, sigma, num, batch_size, device):
    """
    Args:
        - x (torch.Tensor): Single input image of size [channel x width x height]
        - sigma (float): Standard deviation of Gaussian noise
        - c (int): Coordinate of the feature vector to certify
        - num (int): Number of samples for Monte Carlo sampling
        - batch_size (int): batch size
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    x = x.to(device)
    with torch.no_grad():
        feature_vecs = []
        x = x.sub(mean[None, :, None, None]).div(std[None, :, None, None])
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device=device) * sigma
            images = batch + noise
            _, predictions = model2(images, feature = True)
            feature_vecs.extend(predictions.cpu())
           # print(predictions.device)
        return feature_vecs


def estimated_qu_ql(eps, sample_count, sigma, conf_thres=.99999):
    theo_perc_u = stats.norm.cdf(eps / sigma)
    theo_perc_l = stats.norm.cdf(-eps / sigma)

    q_u_u = sample_count + 1
    q_u_l = math.ceil(theo_perc_u * sample_count)
    q_l_u = math.floor(theo_perc_l * sample_count)
    q_l_l = 0
    q_u_final = q_u_u
    for q_u in range(q_u_l, q_u_u):
        conf = stats.binom.cdf(q_u - 1, sample_count, theo_perc_u)
        if conf > conf_thres:
            q_u_final = q_u
            break
    if q_u_final >= sample_count:
        raise ValueError("Quantile index upper bound is larger than n_samples. Increase n_samples or sigma; "
                         "or reduce eps.")
    q_l_final = q_l_l
    for q_l in range(q_l_u, q_l_l, -1):
        conf = 1-stats.binom.cdf(q_l-1, sample_count, theo_perc_l)
        if conf > conf_thres:
            q_l_final = q_l
            break

    return q_u_final, q_l_final
                                                  
                                                  
def get_ms_accuracy_and_logit_stats(smoothing_method, ms_results, sigma, W, b, device, WN):
    """Returns per class accuracy of baseline model and MS model."""
    
    accs = dict()
    logits_below_threshold = 0
    pred_logits_gap = []
    pert_logits_gap = []
    pred_logits_gap_correct = []
    pert_logits_gap_correct = []
    for img_class in ['All', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        correct_med = []
        correct_model = []
        for img_nr in ms_results.keys():
            if (img_class == 'All') or (img_class != 'All' and ms_results[img_nr]['class'] == img_class):
                y_med = ms_results[img_nr][sigma][0]
                y_med = torch.Tensor(y_med).to(device)
                y_model = ms_results[img_nr]['pred']
                y_model = torch.Tensor(y_model).to(device)
                y_pert = ms_results[img_nr]['z_pert']
                y_pert = torch.Tensor(y_pert).to(device)
                out_med = W@y_med + b
                out_model = W@y_model + b
                out_pert = W@y_pert + b
                pred_med = torch.argmax(out_med)
                pred_model = torch.argmax(out_model)
                label = ms_results[img_nr]['class']
                correct_med.append(int(pred_med) == label)
                correct_model.append(int(pred_model) == label)
                ms_results[img_nr]['correctly_predicted'] = int(pred_model) == label
                                                  
                if img_class == 'All':
                    # logit stats
                    out_model_gap = torch.sort(out_model, descending=True).values[0] - torch.sort(out_model, 
                                                                                                   descending=True).values[1]
                    ms_results[img_nr]['logit_gap'] = out_model_gap
                    out_pert_gap = torch.sort(out_pert, descending=True).values[0] - torch.sort(out_pert, 
                                                                                                   descending=True).values[1]
                    pred_logits_gap.append(out_model_gap.detach().cpu().numpy())
                    pert_logits_gap.append(out_pert_gap.detach().cpu().numpy())

                    out_model_gap_correct = abs(torch.sort(out_model, descending=True).values[0] - torch.sort(out_model, 
                                                                                                   descending=True).values[ms_results[img_nr]['class']])
                    out_pert_gap_correct = abs(torch.sort(out_pert, descending=True).values[0] - torch.sort(out_pert, 
                                                                                                   descending=True).values[ms_results[img_nr]['class']])
                    pred_logits_gap_correct.append(out_model_gap_correct.detach().cpu().numpy())
                    pert_logits_gap_correct.append(out_pert_gap_correct.detach().cpu().numpy())

                    if WN:
                        if smoothing_method == 'MS':
                            print(type(torch.norm(out_model - out_pert, p=np.inf).detach().cpu().numpy()), 
                                  type(ms_results[img_nr]['l2_bounds_norm']))
                            if torch.norm(out_model - out_pert, p=np.inf).detach().cpu().numpy() <= ms_results[img_nr]['l2_bounds_norm']:
                                logits_below_threshold += 1
                        elif smoothing_method == 'CS' and torch.norm(out_model - out_pert, p=np.inf).detach().cpu().numpy() <= ms_results[img_nr]['bound']:
                            logits_below_threshold += 1
                                                  
        logits_stats = {'pred': {'min': float(np.min(pred_logits_gap)), 
                          'mean': float(np.mean(pred_logits_gap)), 
                          'max': float(np.max(pred_logits_gap)), 
                          'std': float(np.std(pred_logits_gap))},
                   'pert': {'min': float(np.min(pert_logits_gap)), 
                            'mean': float(np.mean(pert_logits_gap)), 
                            'max': float(np.max(pert_logits_gap)), 
                            'std': float(np.std(pert_logits_gap))}}
        logits_stats_df = pd.DataFrame(logits_stats).T
        
        logits_stats_correct = {'pred': {'min': float(np.min(pred_logits_gap_correct)), 
                          'mean': float(np.mean(pred_logits_gap_correct)), 
                          'max': float(np.max(pred_logits_gap_correct)), 
                          'std': float(np.std(pred_logits_gap_correct))},
                   'pert': {'min': float(np.min(pert_logits_gap_correct)), 
                            'mean': float(np.mean(pert_logits_gap_correct)), 
                            'max': float(np.max(pert_logits_gap_correct)), 
                            'std': float(np.std(pert_logits_gap_correct))}}
        logits_stats_correct_df = pd.DataFrame(logits_stats_correct).T        
                                                  
        accs[str(img_class)] = {'base_model': np.mean(correct_model), 
                                'ms_model': np.mean(correct_med)}
        accs_df = pd.DataFrame(accs)

    if WN:
        print("max_i |logits(z)_i - logits(z')_i| <= ||z_upper - z_lower||_2 for {}/{}".format(logits_below_threshold,
                                                                                           len(list(ms_results.keys()))))
                                  
    return ms_results, accs_df, logits_stats_df, logits_stats_correct_df


def theoretical_adversary(model, x, y, eps_range):
    """ Computes the theoretical lower bound on adversarial accuracy. """
    logits = model(x)
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, y.view(-1, 1), 1)
    correct_logit = (logits * one_hot).sum(1)
    worst_wrong_logit = logits[one_hot == 0].view(one_hot.size(0), -1).max(1)[0]

    adv_margin = F.relu(correct_logit - worst_wrong_logit)

    eps_acc = []
    for e in eps_range:
        eps_acc.append(np.expand_dims((adv_margin > 2 * e).float().detach().cpu().numpy(), 0))
    return np.concatenate(eps_acc, 0).T


def theoretical_adversary_ms(logits, y, bounds_norm):
    """ Computes the theoretical lower bound on adversarial accuracy. """
    one_hot = torch.zeros_like(logits)
    one_hot.scatter_(1, y.view(-1, 1), 1)
    correct_logit = (logits * one_hot).sum(1)
    worst_wrong_logit = logits[one_hot == 0].view(one_hot.size(0), -1).max(1)[0]

    adv_margin = F.relu(correct_logit - worst_wrong_logit)

    eps_acc = []
    eps_acc.append(np.expand_dims((adv_margin > 2 * bounds_norm).float().detach().cpu().numpy(), 0))
    return np.concatenate(eps_acc, 0).T


def get_theoretical_lower_bound(model, data_loader, l_constant, eps_range, device):

    # define mean and std for preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    model.eval()

    with torch.no_grad():
        eps_acc = []

        for images, targets in tqdm(data_loader):
            images = images.to(device)
            targets = targets.to(device)
            # manual preprocessing
            images = images.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
            
            eps_acc.append(theoretical_adversary(model, images, targets, l_constant * eps_range))
            
        avg_eps_acc = np.concatenate(eps_acc, 0).mean(0)

        results = {'eps': eps_range.tolist(),
                   'acc': avg_eps_acc.tolist()}

    return results


def get_robust_accuracy_bounds(model_name_run, k, ms_results_dict, theoretical_accuracies, max_nr_img):
    ms_results = ms_results_dict[model_name_run]
    certifiably_correct = []
    k_upper_bound = []
    nr_correctly_predicted = 0
    for img_nr in ms_results.keys():
        if ms_results[img_nr]['correctly_predicted']:
            if 'WN' in model_name_run:
                bounds_norm = ms_results[img_nr]['l2_bounds_norm']
            elif 'LC' in model_name_run:
                bounds_norm = ms_results[img_nr]['linf_bounds_norm']
            logit_gap = ms_results[img_nr]['logit_gap']#.detach().cpu().numpy()
            k_upper_bound.append(1/2*logit_gap/torch.from_numpy(bounds_norm))
            nr_correctly_predicted += 1
            if logit_gap >= 2 * k * bounds_norm:
                certifiably_correct.append(img_nr)

    print('\nWorking on ', model_name_run, 'using k =', k)
    print('\nOut of {} images, {} are certifiably robust for perturbations with an l2-norm < eps.'.format(nr_correctly_predicted,
                                                                                                           len(certifiably_correct)))
    print('Unperturbed accuracy: {}%\nCertifiable accuracy: {}%'.format(nr_correctly_predicted/max_nr_img*100,
                                                                        len(certifiably_correct)/max_nr_img*100))

    upper_bound_k = np.around(torch.median(torch.stack(k_upper_bound)).detach().cpu().numpy(),3)
    print(f'Would need k <= {upper_bound_k}, to have at least 50% of samples certifiably correct.')

    upper_bound_k = np.around(np.quantile(torch.stack(k_upper_bound).detach().cpu().numpy(), 0.2),3)
    print(f'Would need k <= {upper_bound_k}, to have at least 80% of samples certifiably correct.')

    if model_name_run in theoretical_accuracies:
        print('\nLC - Theoretical accuracy:')
        display(pd.DataFrame(theoretical_accuracies[model_name_run]).set_index(['eps']).T)


def get_robustness_bound_multiple_models(model_name_k_dict, ms_results_dict, theoretical_accuracies, max_nr_img):
    for model_name_run in model_name_k_dict.keys():
        set_k = model_name_k_dict[model_name_run]['set_k']
        actual_k = model_name_k_dict[model_name_run]['actual_k']
        get_robust_accuracy_bounds(model_name_run, set_k, ms_results_dict, theoretical_accuracies, max_nr_img)
        if set_k != actual_k:#
            print(f'Set k: {set_k} - Actual k: {actual_k}')
            get_robust_accuracy_bounds(model_name_run, actual_k, ms_results_dict, theoretical_accuracies, max_nr_img)


def spectral_norm(weight, n_power_iterations=1, eps=1e-12):
    
    weight_mat = weight
    h, w = weight_mat.size()
    u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=eps)
    v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=eps)

    with torch.no_grad():
        for _ in range(n_power_iterations):
            v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=eps, out=v)
            u = normalize(torch.mv(weight_mat, v), dim=0, eps=eps, out=u)
        if n_power_iterations > 0:
            u = u.clone(memory_format=torch.contiguous_format)
            v = v.clone(memory_format=torch.contiguous_format)

    sigma = torch.dot(u, torch.mv(weight_mat, v))
    return sigma

def get_k_of_trained_model(model_info, device):
    # return dict mapping model_name_run to specified k and actual k
    # pass that dict to get_robust_accuracy_bounds --> if specified_k != actual_k --> Return bound for both
    model_name_k_dict = {}
    for model_name in model_info:
        for model_run in [mv for mv in model_info[model_name].keys() if mv != 'fc_layers']:
            model_name_run = f'{model_name} - {model_run}'
            fc_layers = model_info[model_name]['fc_layers']
            # determine set k for given model
            if fc_layers[0] == 'WN':
                set_k = fc_layers[2]
            elif fc_layers[0] == 'LC':
                set_k = fc_layers[1]['l_constant']
            # determine actual k for given model
            if Path(model_info[model_name][model_run]).exists(): # if model file is currently stored on server
                checkpoint = torch.load(model_info[model_name][model_run]) 
                if fc_layers[0] == 'WN' and 'fc.bias' in checkpoint.keys():
                    checkpoint['fc.0.bias'] = checkpoint.pop('fc.bias')
                    checkpoint['fc.0.weight_g'] = checkpoint.pop('fc.weight_g')
                    checkpoint['fc.0.weight_v'] = checkpoint.pop('fc.weight_v')
                model = wideresnet(layers=32, widening_factor = 10, num_classes = 10, fc_layers=fc_layers).to(device)
                model.load_state_dict(checkpoint)
                model.eval()
                
                actual_k_per_layer = []
                for layer in range(len(fc_layers[1]['layer_sizes'])):
        
                    if fc_layers[0] == 'WN':
                        # get layer's k
                        g = model.fc[2*layer].weight_g
                        actual_k_layer = g.max()                                       
                        
                    elif fc_layers[0] == 'LC':
                        # get layer's lipschitz constant
                        W = model.fc[3*layer].weight
                        scaling_factor = model.fc[3*layer+1].factor
                        print('Scaling factor:', scaling_factor)
                        actual_k_layer = abs(W).sum(dim=1).max() * scaling_factor

                    actual_k_per_layer.append(actual_k_layer.detach().cpu().numpy())
                
                actual_k = np.prod(actual_k_per_layer)
                
            model_name_k_dict[model_name_run] = {'actual_k': actual_k,
                                                    'set_k': set_k}
            
    return model_name_k_dict


def compare_weight_stats(WB_model_info, device):
    wb_stats = {}
    for i, model_nr in enumerate(WB_model_info.keys()):

        fc_layers = WB_model_info[model_nr]['fc_layers']

        for model_name in [model_name for model_name in WB_model_info[model_nr].keys() if model_name != 'fc_layers']:

            # Get Base Model
            model = wideresnet(layers=32, widening_factor = 10, num_classes = 10, 
                              fc_layers=fc_layers).to(device)
            checkpoint = torch.load(WB_model_info[model_nr][model_name]) 
            model.load_state_dict(checkpoint)

            if fc_layers[0] == 'WN':
                g = model.fc[0].weight_g
                v = model.fc[0].weight_v 
                norm_v = torch.norm(v, dim = 1).reshape((-1,1))
                W = g* v/norm_v
                b = model.fc[0].bias

            elif fc_layers[0] == 'LC':
                W = model.fc[0].weight
                b = model.fc[0].bias

            else:
                W = model.fc.weight
                b = model.fc.bias
                
            W_L2_row_norms = [np.around(float(torch.norm(W, dim = 1)[i]), 2) for i in range(10)]

            wb_stats[model_nr + ' ' + model_name] = {'W_min': np.around(float(torch.min(W)), 2),
                                                     'W_mean': np.around(float(torch.mean(W)), 2),
                                                     'W_std': np.around(float(torch.std(W)), 2),
                                                     'W_max': np.around(float(torch.max(W)), 2),
                                                     'W_L2_r0': np.around(float(torch.norm(W, dim = 1)[0]), 2),
                                                     'W_L2_r1': np.around(float(torch.norm(W, dim = 1)[1]), 2),
                                                     'W_L2_r2': np.around(float(torch.norm(W, dim = 1)[2]), 2),
                                                     'W_L2_r3': np.around(float(torch.norm(W, dim = 1)[3]), 2),
                                                     'W_L2_r4': np.around(float(torch.norm(W, dim = 1)[4]), 2),
                                                     'W_L2_r5': np.around(float(torch.norm(W, dim = 1)[5]), 2),
                                                     'W_L2_r6': np.around(float(torch.norm(W, dim = 1)[6]), 2),
                                                     'W_L2_r7': np.around(float(torch.norm(W, dim = 1)[7]), 2),
                                                     'W_L2_r8': np.around(float(torch.norm(W, dim = 1)[8]), 2),
                                                     'W_L2_r9': np.around(float(torch.norm(W, dim = 1)[9]), 2),
                                                     
                                                     'W_L2_row_min': np.around(np.min(W_L2_row_norms),2),
                                                     'W_L2_row_mean': np.around(np.mean(W_L2_row_norms),2),
                                                     'W_L2_row_std': np.around(np.std(W_L2_row_norms),2),
                                                     'W_L2_row_max': np.around(np.max(W_L2_row_norms),2),
                                                     
                                                     'W_inf_norm': np.around(float(abs(W).sum(dim=1).max()), 2),
                                                     'b_min': np.around(float(torch.min(b)), 2),
                                                     'b_mean': np.around(float(torch.mean(b)), 2),
                                                     'b_std': np.around(float(torch.std(b)), 2),
                                                     'b_max': np.around(float(torch.max(b)), 2),
                                                     'b_L2_norm': np.around(float(torch.norm(b)), 2),
                                                     'b_Linf_norm': np.around(float(torch.norm(b, p = np.inf)), 2)}

    wb_stats_df = pd.DataFrame(wb_stats)
    return wb_stats_df


def run_MS_for_TL_models_ml(smoothing_method, model_info, loaders, n, eps, sigma, alpha, 
                            batch_size, max_nr_img, attack_params, nr_outputs, device):
    """ Locally rewrite get_ms_accuracy_and_logit_stats for mulltiple layers. """
    ms_results_dict = {}
    theoretical_accuracies = {}
    eps_range = eps_range = np.array([0.001, 0.01, 0.0313, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]) 
                                                       
    for model_name in model_info:
        fc_layers = model_info[model_name]['fc_layers']
        if fc_layers[0] == 'LC':
            l_constant = fc_layers[1]['l_constant']
        else:
            l_constant = False      
        for model_run in [mv for mv in model_info[model_name].keys() if mv != 'fc_layers']:
            
            print('\nWorking on model {} - {}'.format(model_name, model_run))
                
            data_loader = loaders[model_run]
            if Path(model_info[model_name][model_run]).exists(): # if model file is currently stored on server
                # Get Base Model
                checkpoint = torch.load(model_info[model_name][model_run])
                if fc_layers[0] == 'WN' and 'fc.bias' in checkpoint.keys():
                    checkpoint['fc.0.bias'] = checkpoint.pop('fc.bias')
                    checkpoint['fc.0.weight_g'] = checkpoint.pop('fc.weight_g')
                    checkpoint['fc.0.weight_v'] = checkpoint.pop('fc.weight_v')
                
                model = wideresnet(layers=32, widening_factor = 10, num_classes = 10, fc_layers=fc_layers).to(device)
                model.load_state_dict(checkpoint)
                model.eval()
                
                display(pd.DataFrame(get_per_class_accuracies(model, data_loader, device), index = ['Accuracy']))
    
                if smoothing_method == 'MS': 
                    ms_results, avg_eps_acc = run_median_smoothing_single_sigma(n, eps, sigma, alpha, 
                                                                                model, data_loader, batch_size, 
                                                                                max_nr_img, attack_params, 
                                                                                nr_outputs, l_constant, device)
                
                elif smoothing_method == 'CS':
                    avg_eps_acc = None
                    ms_results = run_center_smoothing(eps, sigma, model, data_loader, batch_size, 
                                                      max_nr_img, attack_params, nr_outputs, device)
                        
                # Print stats on ||z_upper - z_lower|| and ||z - z'||
                ms_results, tightness_df_bounds, tightness_df_pert = run_MS_TL_evaluations_norms(smoothing_method, 
                                                                                                 ms_results, sigma)
                print('\nStats on tightness of bounds: ||z_upper - z_lower||:')
                display(tightness_df_bounds)
                
                print("\nStats on ||z - z'||:")
                display(tightness_df_pert)
                
                # Print stats on logits
                ms_results, accs_df, logits_stats_df, logits_stats_correct_df = \
                    get_ms_accuracy_and_logit_stats_ml(smoothing_method, ms_results, sigma, model, device, fc_layers[0]=='WN')
                
                print('\nMS Accuracies:')
                display(accs_df)
                
                print('\nStats on logits:')
                display(logits_stats_df)
                
                print('\nStats on logits - Correct:')
                display(logits_stats_correct_df)
                
                if fc_layers[0] == 'LC':
                    l_constant = fc_layers[1]['l_constant']
                    theoretical_accuracies[model_name + ' - ' + model_run] = get_theoretical_lower_bound(model, data_loader, 
                                                                                                         l_constant, eps_range, 
                                                                                                         device)
                ms_results_dict[model_name + ' - ' + model_run] = ms_results
            else:
                print('Path does not exist!')
    return ms_results_dict, theoretical_accuracies, avg_eps_acc

def get_ms_accuracy_and_logit_stats_ml(smoothing_method, ms_results, sigma, model, device, WN):
    """Returns per class accuracy of baseline model and MS model."""
    
    accs = dict()
    logits_below_threshold = 0
    pred_logits_gap = []
    pert_logits_gap = []
    pred_logits_gap_correct = []
    pert_logits_gap_correct = []
    with torch.no_grad():
        for img_class in ['All', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            correct_med = []
            correct_model = []
            for img_nr in ms_results.keys():
                if (img_class == 'All') or (img_class != 'All' and ms_results[img_nr]['class'] == img_class):
                    y_med = ms_results[img_nr][sigma][0]
                    y_med = torch.Tensor(y_med).to(device)
                    y_model = ms_results[img_nr]['pred']
                    y_model = torch.Tensor(y_model).to(device)
                    y_pert = ms_results[img_nr]['z_pert']
                    y_pert = torch.Tensor(y_pert).to(device)
                    out_med = model.fc(y_med.reshape(1, -1)).flatten() #W@y_med + b
                    out_model = model.fc(y_model.reshape(1, -1)).flatten() #W@y_model + b
                    out_pert = model.fc(y_pert.reshape(1, -1)).flatten() # W@y_pert + b
                    pred_med = torch.argmax(out_med)
                    pred_model = torch.argmax(out_model)
                    label = ms_results[img_nr]['class']
                    correct_med.append(int(pred_med) == label)
                    correct_model.append(int(pred_model) == label)
                    ms_results[img_nr]['correctly_predicted'] = int(pred_model) == label

                    if img_class == 'All':
                        # logit stats
                        out_model_gap = torch.sort(out_model, descending=True).values[0] - torch.sort(out_model, 
                                                                                                       descending=True).values[1]
                        ms_results[img_nr]['logit_gap'] = out_model_gap
                        out_pert_gap = torch.sort(out_pert, descending=True).values[0] - torch.sort(out_pert, 
                                                                                                       descending=True).values[1]
                        pred_logits_gap.append(out_model_gap.detach().cpu().numpy())
                        pert_logits_gap.append(out_pert_gap.detach().cpu().numpy())

                        out_model_gap_correct = abs(torch.sort(out_model, descending=True).values[0] - torch.sort(out_model, 
                                                                                                       descending=True).values[ms_results[img_nr]['class']])
                        out_pert_gap_correct = abs(torch.sort(out_pert, descending=True).values[0] - torch.sort(out_pert, 
                                                                                                       descending=True).values[ms_results[img_nr]['class']])
                        pred_logits_gap_correct.append(out_model_gap_correct.detach().cpu().numpy())
                        pert_logits_gap_correct.append(out_pert_gap_correct.detach().cpu().numpy())

                        if WN:
                            if smoothing_method == 'MS' and torch.norm(out_model - out_pert, p=np.inf).detach().cpu().numpy() <= ms_results[img_nr]['l2_bounds_norm']:
                                logits_below_threshold += 1
                            elif smoothing_method == 'CS' and torch.norm(out_model - out_pert, p=np.inf).detach().cpu().numpy() <= ms_results[img_nr]['bound']:
                                logits_below_threshold += 1
                                                  
            logits_stats = {'pred': {'min': float(np.min(pred_logits_gap)), 
                              'mean': float(np.mean(pred_logits_gap)), 
                              'max': float(np.max(pred_logits_gap)), 
                              'std': float(np.std(pred_logits_gap))},
                       'pert': {'min': float(np.min(pert_logits_gap)), 
                                'mean': float(np.mean(pert_logits_gap)), 
                                'max': float(np.max(pert_logits_gap)), 
                                'std': float(np.std(pert_logits_gap))}}
            logits_stats_df = pd.DataFrame(logits_stats).T

            logits_stats_correct = {'pred': {'min': float(np.min(pred_logits_gap_correct)), 
                              'mean': float(np.mean(pred_logits_gap_correct)), 
                              'max': float(np.max(pred_logits_gap_correct)), 
                              'std': float(np.std(pred_logits_gap_correct))},
                       'pert': {'min': float(np.min(pert_logits_gap_correct)), 
                                'mean': float(np.mean(pert_logits_gap_correct)), 
                                'max': float(np.max(pert_logits_gap_correct)), 
                                'std': float(np.std(pert_logits_gap_correct))}}
            logits_stats_correct_df = pd.DataFrame(logits_stats_correct).T        

            accs[str(img_class)] = {'base_model': np.mean(correct_model), 
                                    'ms_model': np.mean(correct_med)}
            accs_df = pd.DataFrame(accs)

        if WN:
            print("max_i |logits(z)_i - logits(z')_i| <= ||z_upper - z_lower||_2 for {}/{}".format(logits_below_threshold,
                                                                                               len(list(ms_results.keys()))))
                                  
    return ms_results, accs_df, logits_stats_df, logits_stats_correct_df
