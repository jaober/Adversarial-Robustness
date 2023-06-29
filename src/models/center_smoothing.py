"""
Implements Center Smoothing.
Main source of underlying logic: https://arxiv.org/pdf/2102.09701.pdf
"""

import torch
from math import ceil
import pandas as pd
import numpy as np

def add_gaussian_noise(model2, x, sigma, num, batch_size, device):
    """Returns feature vectors obtained by applying the model to noisy input data
    Input:
        model2: model that includes preprocessing and returns pre-FC-layer feature vectors
        x (torch.Tensor): Single input image of size [channel x width x height]
        sigma (float): Standard deviation of Gaussian noise
        num (int): Number of samples for Monte Carlo sampling
        batch_size (int): batch size
        device: device
    Output:
        feature_vecs: model(x + N(0, sigma²I))
    """
    x = x.to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    with torch.no_grad():
        feature_vecs = []
        x = x.sub(mean[None, :, None, None]).div(std[None, :, None, None])
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch, device=device) * sigma
            out, predictions = model2(batch + noise, feature = True)#.cpu()
            feature_vecs.extend(predictions)
        feature_vecs = torch.stack(feature_vecs, dim = 0)
        return feature_vecs
    

def get_beta_MEB(Z, device, ratio=1/2):
    """ Returns z in Z that has the minimum median distance from all points in the set (including itself)
        Input:
            Z (torch.Tensor): model(x + N(0, sigma²I))
        Output:
            z: center of ball that contains at least half the points in Z
            r: radius of ball that contains at least half the points in Z
    """
    with torch.no_grad():
        # flatten Z
        Z = Z.to(device)
        Z_flat = torch.flatten(Z, start_dim=1)

        # for all points in Z: Get distances to all other points
        dists = torch.cdist(Z_flat,Z_flat)#.to(device)

        # Get medians of distances
        median_dists = torch.median(dists, dim = 0).values

        # Get minimum median
        argmin_median_dist = torch.argmin(median_dists)

        # Return corresponding z
        z = Z[argmin_median_dist]

        # get radius --> check this again
        r = median_dists[argmin_median_dist]
    
    return z, r


def center_smooth(x, model, sigma, delta, alpha1, n, num_steps, batch_size, device):
    """Estimates center smoothed function
        Source: Center Smoothing paper
        Inputs:
            - x: input image
            - model: model that includes all preprocessing steps and returns the pre-FC-layer feature vectors
            - sigma: standard deviation for Gaussian nois
            - delta: ?? in [0, 1/2]
            - alpha1: in [0,1], should be small
            - n: number of samples to draw
            - num_steps: number of tries for computing the center
            - batch_size: batch_size
            - device: device
        Output: 
            - z: Output of applying estimated center smoothed function to the input image
    """
    x = x.to(device)
    for i in range(num_steps):
        # Get z
        Z = add_gaussian_noise(model, x, sigma, n, batch_size, device).to(device)
        alpha1 = torch.Tensor([alpha1]).to(device)
        delta1 = torch.sqrt(torch.log(2/alpha1)/(2*n))#.to(device)
        
        # Compute z = beta-MEB
        z, r = get_beta_MEB(Z, device)
        # Resample Z
        Z = add_gaussian_noise(model, x, sigma, n, batch_size, device).to(device)
        
        # [TODO] Get p_delta1
        p_delta1 = torch.sum(torch.norm(Z-z,dim=1) <= r)/n #1-torch.exp(-2*n*delta1**2)
        
        delta2 = 1/2 - p_delta1
        
        if delta >= max(delta1, delta2):
            return z
        if i == num_steps-1:
            print('No z found.')
            return None
    
    
def center_certify(model, x, sigma, m, n, eps1, delta, alpha1, alpha2, beta, num_steps, batch_size, device):
    """ Certifies input image via Center Smoothing
    
        Input:
            model: WideResNet_gurobi model, inl. preprocessing, returns feature vectors
            x: random input image
            sigma: Start with 0.1 (--> model trained on 0.1)
            m: number of samples to generate certificates, paper: 10⁶
            n: number of samples used to estimate the smoothed function, paper: 10⁴
            eps1: bound on input perturbation, Paper: eps = h*sigma, eps $\in$ {0.1, 0.2, 0.3, 0.4, 0.5}, h $\in$ {1, 1.5, 2}
            delta: Paper: 0.05
            alpha1: small, in [0,1], Paper: 0.01 
            alpha2: small, in [0,1], Paper: 0.01 
            beta: paper: sprt(ln(300)/c) --> 2??
            num_steps: 10 (paper) 
            batch_size: 128
            device: device
    
        Output:
            eps2: bound on change in output space
    """
    z = center_smooth(x, model, sigma, delta, alpha1, n, num_steps, batch_size, device).to(device)
    if z != None:
        # Sample Z
        Z = add_gaussian_noise(model, x, sigma, n, batch_size, device).to(device)
        # Compute R_tilde
        R_tilde = torch.norm(Z-z, dim=1)
        # Set p 
        standard_gaussian = torch.distributions.normal.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        p_icdf = standard_gaussian.icdf(1/2 + torch.Tensor([delta]))
        p = standard_gaussian.cdf(p_icdf + eps1/sigma)
        # Set q
        q = p + torch.sqrt(torch.log(1/torch.Tensor([alpha2]))/(2*m))
        # Set R_hat
        q = q.to(device)
        R_hat = torch.quantile(R_tilde, q)
        # Set eps2
        eps2 = (1 + beta) * R_hat
        return eps2, z
    else:
        return None, None
    
    
def eval_center_smoothing(cs_results):
    """Prints brief summary of Center Smoothing Dertification.
    Input: 
        cs_results: dictionary containing results of Center Smoothing certification, mapping image number to eps2 value
    Output:
        cs_stats_df: Dataframe containing minimum, mean and maximum of eps2 results
    """
    cs_df = pd.DataFrame(cs_results,index = ['eps2']).T
    
    # For how many images has CS certification succeded?
    cs_df_success = cs_df.loc[cs_df.eps2 != None]
    print('CS Certification successful for {}/{} images.'.format(len(cs_df_success), len(cs_df)))

    # Get stats on eps2 for the successful certifications
    print('\nCS Certification Stats (on {} images):'.format(len(cs_df_success)))
    cs_stats = {'eps2': {'max': np.max(cs_df_success.eps2),
                         'mean': np.mean(cs_df_success.eps2),
                         'min': np.min(cs_df_success.eps2),}}
    
    cs_stats_df = pd.DataFrame(cs_stats).T
    print(cs_stats_df)

    return cs_stats_df    
