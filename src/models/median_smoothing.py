"""
Implements median smoothing, as proposed by Ping-yeh Chiang, Michael Curry, Ahmed Abdelkader, Aounon Kumar,
John Dickerson, and Tom Goldstein. Detection as regression: Certified object detection with median smoothing.
In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing
Systems, volume 33, pages 1275–1286. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/
0dd1bc593a91620daecf7723d2235624-Paper.pdf.
"""
import gurobipy as gp
from gurobipy import GRB
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import pandas as pd
import time
import itertools
from itertools import chain
import collections
from torch.distributions.normal import Normal as standard_gaussian
from scipy.special import binom, comb
from math import ceil
import torch
from torch import nn
from scipy import stats
import math
import json
from tqdm.auto import tqdm
import sys
import os
from IPython.display import display, HTML
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"/ARTL")
from models import wideresnet, WideResNet_gurobi
from data_loader import read_dataset, get_loader
from optimization_pipeline import *


# 
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
        #print('qu: {} \nql: {}'.format(qu, ql))
        y_hat = add_gaussian_noise_ms(model, x, sigma, n, batch_size, device)
        y_hats = torch.stack(y_hat, axis = 0) #torch.cat(y_hat, axis=1)
        # Sort each coordinate along the batch dimension -> Why? Needed?
        y_hat_sorted = torch.sort(y_hats, dim=0)[0]
        #print('y_hat_sorted:', y_hat_sorted.device)
        y_median = torch.median(y_hats, axis = 0)[0]
        #print('y_median:', y_median.device)
        # Take the qth order statistics
        yu = y_hat_sorted[qu,:]
        yl = y_hat_sorted[ql,:]
        #print('yu:', yu.device)
        #print('yl:', yl.device)
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


def run_median_smoothing(n, sigmas, alpha, batch_size, json_path, model_path,
                         num_classes, mode, data_dir, num_workers, max_nr_img, fc_layers):

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('working on', device)

    # Load model
    if type(model_path) == str:
        # path to saved model parameters was passed as input
        model = WideResNet_gurobi(layers=32, widening_factor=10, num_classes=num_classes, fc_layers = fc_layers).to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
    else:
        # model was passed as input
        model = model_path

    model.eval()

    # load data
    img_transforms = transforms.ToTensor()

    img_loader = get_loader(mode, data_dir, transform=img_transforms, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, max_nr_img=max_nr_img)


    qu, ql = estimated_qu_ql(sigmas[0], n, sigmas[0], conf_thres=alpha)
    print('Working with qu: {}, ql: {}'.format(qu, ql))

    ms_results = {}
    img_nrs = 0
    print('Iterating over {} batches.'.format(max_nr_img//batch_size + int(max_nr_img%batch_size > 0)))
    with torch.no_grad():
        for images, labels in tqdm(img_loader):
            images = images.to(device)
            predictions = model(images)

            print('Iterating over {} images in batch.'.format(len(images)))
            for img_nr in tqdm(range(len(images))):
                x = images[img_nr]
                ms_results[img_nrs + img_nr] = dict()
                ms_results[img_nrs + img_nr]['pred'] = predictions[img_nr].detach().cpu().numpy()
                ms_results[img_nrs + img_nr]['class'] = int(labels[img_nr])

                for sigma in sigmas:
                    eps = sigma
                    y_med, yl, yu = certify(model, x, qu, ql, sigma, n, eps, alpha, batch_size, device)
                    ms_results[img_nrs + img_nr][sigma] = [y_med.detach().cpu().numpy(), 
                                                           yl.detach().cpu().numpy(), 
                                                           yu.detach().cpu().numpy()]
            img_nrs += len(images)
        

    return ms_results


def visualize_ms_bounds(y_median, y_lower, y_upper, sigma):
    """Visualize MS bounds for one image only."""
    
    y_median_sorted, sorted_indices = torch.sort(y_median)
    plt.figure(figsize=(15, 10))
    plt.plot(range(len(y_median)), y_median_sorted, label = 'median')
    plt.plot(range(len(y_lower)), y_lower[sorted_indices], label = 'lower bound')
    plt.plot(range(len(y_upper)), y_upper[sorted_indices], label = 'upper bound')
    plt.legend()
    plt.title('Median Smoothing Bounds (Sorted), $\sigma = {}$'.format(sigma))
    plt.xlabel('Coordinate of feature vector')
    plt.ylabel('Value of feature vector');

    
def get_sigma_summary_for_single_image(ms_results, img_nr, sigmas):
    """Print key stats (e.g. L2-distance between vectors) for different values of 
        sigma for one image only."""
    
    sigma_eval = dict()
    #for img_nr in range(ms_results.keys())
    y = ms_results[img_nr]['pred']
    for sigma in sigmas:
        sigma_eval[sigma] = dict()
        [y_med, yl, yu] =  ms_results[img_nr][sigma]
        sigma_eval[sigma]['yl <= y_median'] = int(np.sum(yl <= y_med))
        sigma_eval[sigma]['yl <= y'] = int(np.sum(yl <= y))
        sigma_eval[sigma]['yu >= y_median'] = int(np.sum(yu >= y_med))
        sigma_eval[sigma]['yu >= y'] = int(np.sum(yu >= y))
        sigma_eval[sigma]['norm(y - yu)'] = round(float(np.linalg.norm(y - yu)),3)
        sigma_eval[sigma]['norm(y - yl)'] = round(float(np.linalg.norm(y - yl)),3)
        sigma_eval[sigma]['norm(yl - y_med)'] = round(float(np.linalg.norm(yu - y_med)),3)
        sigma_eval[sigma]['norm(yu - y_med)'] = round(float(np.linalg.norm(yl - y_med)),3)
        sigma_eval[sigma]['norm(y - y_med)'] = round(float(np.linalg.norm(y - y_med)),3)

    sigma_one_image = pd.DataFrame(sigma_eval)    
    return sigma_one_image


def get_sigma_summary_for_multiple_images(ms_results, sigmas):
    """Print averaged key stats (e.g. L2-distance between vectors) for different 
        values of sigma for one image only."""
    
    sigma_eval = dict()
    for sigma in sigmas:
        sigma_eval[sigma] = dict()
        sigma_img = dict()
        sigma_eval[sigma]['yl <= y_median'] = dict()
        sigma_eval[sigma]['yl <= y'] = dict()
        sigma_eval[sigma]['yu >= y_median'] = dict()
        sigma_eval[sigma]['yu >= y']  = dict()
        sigma_eval[sigma]['norm(y - yu)'] = dict()
        sigma_eval[sigma]['norm(y - yl)'] = dict()
        sigma_eval[sigma]['norm(yl - y_med)']  = dict()
        sigma_eval[sigma]['norm(yu - y_med)']  = dict()
        sigma_eval[sigma]['norm(y - y_med)'] = dict()
        for img_nr in ms_results.keys():
            y = ms_results[img_nr]['pred']
            [y_med, yl, yu] =  ms_results[img_nr][sigma]
            sigma_eval[sigma]['yl <= y_median'][img_nr] = int(np.sum(yl <= y_med))
            sigma_eval[sigma]['yl <= y'][img_nr] = int(np.sum(yl <= y))
            sigma_eval[sigma]['yu >= y_median'][img_nr] = int(np.sum(yu >= y_med))
            sigma_eval[sigma]['yu >= y'][img_nr] = int(np.sum(yu >= y))
            sigma_eval[sigma]['norm(y - yu)'][img_nr] = round(float(np.linalg.norm(y - yu)),3)
            sigma_eval[sigma]['norm(y - yl)'][img_nr] = round(float(np.linalg.norm(y - yl)),3)
            sigma_eval[sigma]['norm(yl - y_med)'][img_nr] = round(float(np.linalg.norm(yu - y_med)),3)
            sigma_eval[sigma]['norm(yu - y_med)'][img_nr] = round(float(np.linalg.norm(yl - y_med)),3)
            sigma_eval[sigma]['norm(y - y_med)'][img_nr] = round(float(np.linalg.norm(y - y_med)),3)
            
        sigma_eval[sigma]['yl <= y_median'] = np.mean(list(sigma_eval[sigma]['yl <= y_median'].values()))
        sigma_eval[sigma]['yl <= y'] = np.mean(list(sigma_eval[sigma]['yl <= y'].values()))
        sigma_eval[sigma]['yu >= y_median'] = np.mean(list(sigma_eval[sigma]['yu >= y_median'].values()))
        sigma_eval[sigma]['yu >= y']  = np.mean(list(sigma_eval[sigma]['yu >= y'].values()))
        sigma_eval[sigma]['norm(y - yu)'] = np.mean(list(sigma_eval[sigma]['norm(y - yu)'].values()))
        sigma_eval[sigma]['norm(y - yl)'] = np.mean(list(sigma_eval[sigma]['norm(y - yl)'].values()))
        sigma_eval[sigma]['norm(yl - y_med)'] = np.mean(list(sigma_eval[sigma]['norm(yl - y_med)'].values()))
        sigma_eval[sigma]['norm(yu - y_med)'] = np.mean(list(sigma_eval[sigma]['norm(yu - y_med)'].values()))
        sigma_eval[sigma]['norm(y - y_med)'] = np.mean(list(sigma_eval[sigma]['norm(y - y_med)'].values()))

    sigma_avg = pd.DataFrame(sigma_eval)    
    return sigma_avg


def get_class_distribution(ms_results):
    """Plots class histogram of MS results."""
    
    class_frequency = dict()
    for img_nr in ms_results.keys():
        if ms_results[img_nr]['class'] in class_frequency.keys():
            class_frequency[ms_results[img_nr]['class']] += 1
        else:
            class_frequency[ms_results[img_nr]['class']] = 1
    class_frequency
    plt.bar(class_frequency.keys(), class_frequency.values())
    plt.title('Number of samples per class')
    plt.xlabel('Class')
    plt.ylabel('Number of samples');
    return class_frequency


def get_ms_accuracy(ms_results, sigma, W, b, device):
    """Returns per class accuracy of baseline model and MS model."""
    
    accs = dict()
    for img_class in ['All', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        correct_med = []
        correct_model = []
        for img_nr in ms_results.keys():
            if (img_class == 'All') or (img_class != 'All' and ms_results[img_nr]['class'] == img_class):
                y_med = ms_results[img_nr][sigma][0]
                y_med = torch.Tensor(y_med).to(device)
                y_model = ms_results[img_nr]['pred']
                y_model = torch.Tensor(y_model).to(device)
                out_med = W@y_med + b
                out_model = W@y_model + b
                pred_med = torch.argmax(out_med)
                pred_model = torch.argmax(out_model)
                label = ms_results[img_nr]['class']
                correct_med.append(int(pred_med) == label)
                correct_model.append(int(pred_model) == label)
        accs[str(img_class)] = {'base_model': np.mean(correct_model), 
                                'ms_model': np.mean(correct_med)}
    return pd.DataFrame(accs)


def get_median_smoothing_summary(ms_results, sigma, num_classes):
    """Returns per-class key stats (e.g. distance of feature vector to bounds)."""
    
    ms_eval = dict()
    img_classes = ['All']
    img_classes.extend(range(num_classes))
    for img_class in img_classes:
        ms_eval[img_class] = dict()
        ms_eval[img_class]['yl <= y_median'] = dict()
        ms_eval[img_class]['yl <= y'] = dict()
        ms_eval[img_class]['yu >= y_median'] = dict()
        ms_eval[img_class]['yu >= y']  = dict()
        ms_eval[img_class]['norm(y - yu)'] = dict()
        ms_eval[img_class]['norm(y - yl)'] = dict()
        ms_eval[img_class]['norm(yl - y_med)']  = dict()
        ms_eval[img_class]['norm(yu - y_med)']  = dict()
        ms_eval[img_class]['norm(y - y_med)'] = dict()
        
    for img_nr in ms_results.keys():
        img_label = ms_results[img_nr]['class']
        y = ms_results[img_nr]['pred']
        [y_med, yl, yu] =  ms_results[img_nr][sigma]
        ms_eval[img_label]['yl <= y_median'][img_nr] = int(np.sum(yl <= y_med))
        ms_eval[img_label]['yl <= y'][img_nr] = int(np.sum(yl <= y))
        ms_eval[img_label]['yu >= y_median'][img_nr] = int(np.sum(yu >= y_med))
        ms_eval[img_label]['yu >= y'][img_nr] = int(np.sum(yu >= y))
        ms_eval[img_label]['norm(y - yu)'][img_nr] = round(float(np.linalg.norm(y - yu)),3)
        ms_eval[img_label]['norm(y - yl)'][img_nr] = round(float(np.linalg.norm(y - yl)),3)
        ms_eval[img_label]['norm(yl - y_med)'][img_nr] = round(float(np.linalg.norm(yu - y_med)),3)
        ms_eval[img_label]['norm(yu - y_med)'][img_nr] = round(float(np.linalg.norm(yl - y_med)),3)
        ms_eval[img_label]['norm(y - y_med)'][img_nr] = round(float(np.linalg.norm(y - y_med)),3)
        ms_eval['All']['yl <= y_median'][img_nr] = int(np.sum(yl <= y_med))
        ms_eval['All']['yl <= y'][img_nr] = int(np.sum(yl <= y))
        ms_eval['All']['yu >= y_median'][img_nr] = int(np.sum(yu >= y_med))
        ms_eval['All']['yu >= y'][img_nr] = int(np.sum(yu >= y))
        ms_eval['All']['norm(y - yu)'][img_nr] = round(float(np.linalg.norm(y - yu)),3)
        ms_eval['All']['norm(y - yl)'][img_nr] = round(float(np.linalg.norm(y - yl)),3)
        ms_eval['All']['norm(yl - y_med)'][img_nr] = round(float(np.linalg.norm(yu - y_med)),3)
        ms_eval['All']['norm(yu - y_med)'][img_nr] = round(float(np.linalg.norm(yl - y_med)),3)
        ms_eval['All']['norm(y - y_med)'][img_nr] = round(float(np.linalg.norm(y - y_med)),3)
        
    for img_class in img_classes:   
        ms_eval[img_class]['yl <= y_median'] = np.mean(list(ms_eval[img_class]['yl <= y_median'].values()))
        ms_eval[img_class]['yl <= y'] = np.mean(list(ms_eval[img_class]['yl <= y'].values()))
        ms_eval[img_class]['yu >= y_median'] = np.mean(list(ms_eval[img_class]['yu >= y_median'].values()))
        ms_eval[img_class]['yu >= y']  = np.mean(list(ms_eval[img_class]['yu >= y'].values()))
        ms_eval[img_class]['norm(y - yu)'] = np.mean(list(ms_eval[img_class]['norm(y - yu)'].values()))
        ms_eval[img_class]['norm(y - yl)'] = np.mean(list(ms_eval[img_class]['norm(y - yl)'].values()))
        ms_eval[img_class]['norm(yl - y_med)'] = np.mean(list(ms_eval[img_class]['norm(yl - y_med)'].values()))
        ms_eval[img_class]['norm(yu - y_med)'] = np.mean(list(ms_eval[img_class]['norm(yu - y_med)'].values()))
        ms_eval[img_class]['norm(y - y_med)'] = np.mean(list(ms_eval[img_class]['norm(y - y_med)'].values()))

    ms_avg = pd.DataFrame(ms_eval)    
    return ms_avg


def get_within_bounds_percentages(ms_results, threshold, class_frequency, sigma, num_classes):
    """Plots percentage of feature vectors for which at least <threshold> 
        coordinates fall between the MS bounds per class."""
    
    nr_within_bounds = dict()
    for img_class in range(num_classes):
        nr_within_bounds[img_class] = 0

    for img_nr in ms_results.keys():
        y = ms_results[img_nr]['pred']
        [y_med, yl, yu] =  ms_results[img_nr][sigma]
        within_bounds_sum = int(np.sum((yl <= y) & (y <= yu))) 
        if within_bounds_sum >= threshold:
            nr_within_bounds[ms_results[img_nr]['class']] += 1

    nr_within_bounds_all = dict()    
    nr_within_bounds_all['All'] = np.sum(list(nr_within_bounds.values()))
    
    for img_class in range(num_classes):    
        nr_within_bounds[img_class] /= class_frequency[img_class]
    
    nr_within_bounds_all['All'] /= np.sum(list(class_frequency.values()))
    nr_within_bounds_all['Per_class'] = nr_within_bounds
    
    plt.figure()
    plt.bar(nr_within_bounds.keys(), nr_within_bounds.values())
    plt.title('Number of samples per class')
    plt.xlabel('Class')
    plt.ylabel('Percentage of samples');
    
    return nr_within_bounds_all


def get_within_bounds_stats(feature_vectors, ms_results, sigma, num_classes):
    nr_within_bounds = dict()
    """Returns minimum, average and maximumnumber of feature vector coordinates 
        within bounds for each class."""

    for img_class in range(num_classes):
        nr_within_bounds[img_class] = []

    for img_nr in ms_results.keys():
        [y_med, yl, yu] =  ms_results[img_nr][sigma]
        if feature_vectors:
            y = feature_vectors[img_nr].detach().cpu().numpy()
        else: 
            y = ms_results[img_nr]['pred']
        within_bounds_sum = int(np.sum((yl <= y) & (y <= yu))) 
        nr_within_bounds[ms_results[img_nr]['class']].extend([within_bounds_sum])

    nr_within_bounds['All'] = list(chain.from_iterable(nr_within_bounds.values()))
    
    nr_within_bounds_all = dict()
    for img_class in nr_within_bounds.keys():    
        nr_within_bounds_all[img_class] = dict()
        
    for img_class in nr_within_bounds.keys():  
        nr_within_bounds_all[img_class]['min'] = int(np.min(nr_within_bounds[img_class]))
        nr_within_bounds_all[img_class]['avg'] = np.around(np.mean(nr_within_bounds[img_class]),2)
        nr_within_bounds_all[img_class]['max'] = int(np.max(nr_within_bounds[img_class]))
    
    within_bounds_stats = pd.DataFrame(nr_within_bounds_all)
    return within_bounds_stats
 
    
def get_bounds_deviation(ms_results, sigma, num_classes):
    """Legacy function, probably rather useless. 
        Returns hardly readable plot of bounds per image class."""
    
    ms_df = pd.DataFrame(ms_results).T
    bounds_deviations = dict()

    for img_class in range(num_classes):
        ms_df_class = ms_df.loc[ms_df['class'] == img_class][sigma]
        ms_df_class_bounds = np.concatenate(ms_df_class.values, axis = 1)
        ms_df_class_bounds = ms_df_class_bounds.reshape((3, 640, len(ms_df_class)))
        
        bounds_deviations[img_class] = collections.defaultdict(dict)
        for i, bound in enumerate(['median', 'lower', 'upper']):
            bounds_deviations[img_class][bound]['min'] = np.min(ms_df_class_bounds[i], axis = 1)
            bounds_deviations[img_class][bound]['max'] = np.max(ms_df_class_bounds[i], axis = 1)
            bounds_deviations[img_class][bound]['avg'] = np.mean(ms_df_class_bounds[i], axis = 1)
    
        # plot bounds deviation
        med_min, med_max, med_avg = bounds_deviations[img_class]['median'].values()
        lower_min, lower_max, lower_avg = bounds_deviations[img_class]['lower'].values()
        upper_min, upper_max, upper_avg = bounds_deviations[img_class]['upper'].values()

        sort_indices = np.argsort(med_avg)
        plt.figure()
        plt.plot(range(640), lower_min[sort_indices], label = 'lower_min', c = 'orange')
        plt.plot(range(640), upper_max[sort_indices], label = 'upper_max', c = 'green')
        plt.plot(range(640), med_avg[sort_indices], label = 'median_avg', c = 'blue')
        plt.fill_between(range(640), med_min[sort_indices], med_max[sort_indices], alpha = 0.3, color = 'blue')

        plt.legend()
        plt.title('Bounds deviation for class {}'.format(img_class))
        plt.xlabel('Sorted coordinates')
        plt.ylabel('Deviation');

    return bounds_deviations



def run_z_optimization_on_ms_results(base_vector, ms_results, X_correct, labels_correct, W, b, device, sigma = 0.1, num_classes=10):
    """Runs z-optimization on MS results. base_vector specifies on which vector to
        run the optimization."""
    
    # Get Tensor of base vectors
    y_selected = {}      
    for img_nr in ms_results.keys():
        if base_vector == 'median':
            y_sel = ms_results[img_nr][sigma][0]
        elif base_vector == 'lower':
            y_sel = ms_results[img_nr][sigma][1]
        elif base_vector == 'upper':
            y_sel = ms_results[img_nr][sigma][2]
        elif base_vector == 'correct':
            y_sel = ms_results[img_nr]['pred']  
        y_selected[img_nr] = torch.from_numpy(y_sel).to(device)
        
    y_selected = torch.stack(tuple(y_selected.values()))

    # Apply z optimization to base vectors
    z_opt_df, opt_feature_vectors, nr_success, nr_total = get_opt_z_incl_report(y_selected, X_correct, 
                                                                                labels_correct, W, b, num_classes, device)

    # Print within bounds stats
    print("Within bounds stats - y_{}' = y_{} + eps.".format(base_vector, base_vector))
    within_bounds_stats = get_within_bounds_stats(opt_feature_vectors, ms_results, sigma=sigma, num_classes=num_classes)
    display(within_bounds_stats)
    return opt_feature_vectors, within_bounds_stats, nr_success, nr_total


def get_opt_z_incl_report(base_feature_vector, X_correct, labels_correct, W, b, num_classes, device):
    """Runs z-optimization on MS results using specified dict of 
        base_feature_vectors"""
    
    z_med_opt_dict = {}
    opt_feature_vectors_med = {}
    for image_nr in range(len(labels_correct)):

        # Fix one sample:
        label = labels_correct[image_nr]
        X_org = X_correct[image_nr].reshape((1,3,32,32)).detach().clone() # preprocessing includedin model2

        z_adv, best_k, best_pred, z_dict = get_optimized_feature_z(base_feature_vector[image_nr], W, b, label, num_classes, device);
        z_med_opt_dict[image_nr] = z_dict
        opt_feature_vectors_med[image_nr] = z_adv

    z_med_opt_df = pd.DataFrame(z_med_opt_dict).T
    
    # Get optimization report
    nr_success, nr_total = get_optimization_report_z(z_med_opt_df)
    
    return z_med_opt_df, opt_feature_vectors_med, nr_success, nr_total


def get_min_abs_distance_to_bounds(y, y_lower, y_upper):
    """Helper function to plot_abs_distance_to_bounds and plot_average_abs_distance_to_bounds.
    
        For each coordinate determine the absolute distance to the nearest bound:
         - 0 if coordinate lies within bounds
         - min(abs(y-yl), abs(y-yu)) otherwise
         - sort vector by values (ascending order)
    """
    abs_distances = np.zeros(len(y))
    for i in range(len(y)):
        if (y[i] >= y_lower[i]) & (y[i] <= y_upper[i]):
            abs_distances[i] = 0
        else:
            abs_distances[i] = min(abs(y[i] - y_lower[i]), abs(y[i] - y_upper[i]))
    abs_distances_sorted = np.sort(abs_distances)
    return abs_distances_sorted


def plot_abs_distance_to_bounds(ys, lower_bound, upper_bound, image_nr, ax):
    """Plots absolute distance of set of feature vectors to to MS bounds per image."""
    
    abs_distances = {}
    
    for y_label in ys.keys():
        abs_distances[y_label] = get_min_abs_distance_to_bounds(ys[y_label], lower_bound, upper_bound)
        ax.plot(range(len(abs_distances[y_label])), abs_distances[y_label], label=y_label)
        
    ax.set_title('Absolute distances to bounds for image {}'.format(image_nr))
    ax.set_xlabel('Coordinates (sorted)')
    ax.set_ylabel('Absolute distance')
    ax.legend();

    
def plot_average_abs_distance_to_bounds(ys, ms_results, sigma=0.1):
    """Plots absolute distance of set of feature vectors to to MS bounds 
        averagedover all image in ms_results."""
        
    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    abs_distances = collections.defaultdict(dict)
    for y_label in ys.keys():
        abs_distances[y_label] = []
    for img_nr in ms_results.keys():
        lower_bound = ms_results[img_nr][sigma][1]
        upper_bound = ms_results[img_nr][sigma][2]
        for y_label in ys.keys():
            abs_distances[y_label].append(get_min_abs_distance_to_bounds(ys[y_label][img_nr], 
                                                                            lower_bound, upper_bound))
    for y_label in abs_distances.keys():
        abs_distances[y_label] = np.mean(abs_distances[y_label], axis = 0)
        ax.plot(range(len(abs_distances[y_label])), abs_distances[y_label], label=y_label)   
        
    ax.set_title('Absolute distances to bounds, averaged over {} images'.format(len(ms_results.keys())))
    ax.set_xlabel('Coordinates (sorted)')
    ax.set_ylabel('Absolute distance')
    ax.legend();
    
    
def get_tightness_stats(ms_results, sigma):
    l2_tightness = []
    linf_tightness = []
    for image_nr in ms_results.keys():
        upper_bound = ms_results[image_nr][sigma][1]
        lower_bound = ms_results[image_nr][sigma][2]
        l2_tightness.append(np.linalg.norm(upper_bound - lower_bound))
        linf_tightness.append(np.linalg.norm(upper_bound - lower_bound, ord = np.inf))
        
    tightness_stats = {'l2': {'min': float(np.min(l2_tightness)), 
                              'mean': float(np.mean(l2_tightness)), 
                              'max': float(np.max(l2_tightness)), 
                              'std': float(np.std(l2_tightness))},
                       'linf': {'min': float(np.min(linf_tightness)), 
                                'mean': float(np.mean(linf_tightness)), 
                                'max': float(np.max(linf_tightness)), 
                                'std': float(np.std(linf_tightness))}}
    tightness_df = pd.DataFrame(tightness_stats).T
    return tightness_df
