"""
Functions used in Robustness Assessment part, e.g. optimization in feature space and optimization in input space
"""
# Imports
import os
import socket
os.environ["GRB_LICENSE_FILE"] = #####
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
import sys
import os
from tqdm.auto import tqdm
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"/ARTL")
from models import wideresnet, WideResNet_gurobi
from data_loader import read_dataset, get_loader


CIFAR10_META_PATH = ######


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_model_weight_norms(models, weight_norms, device):
    """
    models (dict): Maps version name to model path
    device (torch.device): Device to work on
    """
    for model_version, weight_norm in zip(models.keys(), weight_norms):

        # Import trained model
        model_w = wideresnet(layers=32, widening_factor = 10, 
                             num_classes = 10, weight_normalization=weight_norm).to(device)
        checkpoint = torch.load(models[model_version]) 
        model_w.load_state_dict(checkpoint)
        model_w.eval();

        # Get weights of fully connected layer
        W = model_w.fc.weight
        b = model_w.fc.bias
        
        if weight_norm:
            g = model_w.fc.weight_g
            print('\nModel Version: {}'.format(model_version),
                  '\nModel W norm (torch.norm): {}'.format(torch.norm(W)),
                  '\nModel W norm (fc.weight_g): {}'.format(g),
                  '\nModel b norm: {}'.format(torch.norm(b)))
        else:           
            print('\nModel Version: {}'.format(model_version),
                 '\nModel W norm: {}'.format(torch.norm(W)),
                 '\nModel b norm: {}'.format(torch.norm(b)))
        

def get_model(architecture, model_path, device, fc_layers=[None]):
    # Import trained model
    model = architecture(layers=32, widening_factor = 10, num_classes = 10,
                        fc_layers=fc_layers).to(device)
    if device == 'cpu':
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    else: 
        checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval();
    return model


def get_data(max_nr_img, data_dir, model, device, mode='test', batch_size=256, 
             shuffle=True, num_workers=2, debug=False):
    """
    Call e.g. via images, targets, preds, X_correct, z_correct, preds_correct, labels_correct = get_data(100, data_dir)
    """
    # define mean and std for preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # data transformations
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])

    # get testset images
    img_loader = get_loader(mode=mode, 
                            data_dir=data_dir, 
                            transform=img_transforms, 
                            batch_size=batch_size,
                            shuffle=shuffle, 
                            num_workers=num_workers, 
                            max_nr_img=max_nr_img)
    with torch.no_grad():
        for i, (images, targets) in enumerate(img_loader):
            X = images.detach().clone().to(device)
            if debug:
                print('Batch Nr.:', i, '\nShape of Images:', images.shape, '\nShape of Targets:', targets.shape)
                print('Range of Image Values: [{}, {}]'.format(X.min(), X.max()))
            y = targets.to(device)
            # Preprocessing 
            X_prep = X.sub(mean[None, :, None, None]).div(std[None, :, None, None])
            # Apply model and get features z
            out, z = model(X_prep, feature = True)

            # Get model predictions preds
            topk=(1,)
            _, preds = out.topk(max(topk), 1, True, True)
            if debug:
                print(preds)
            preds = preds.t()

            # Filter for correctly predicted inputs X_correct and associated features z_correct
            correct = preds.eq(y.view(1, -1).expand_as(preds)).flatten()
            if debug:
                print(y, correct)
            X_correct, z_correct = images[correct], z[correct]
            preds_correct, labels_correct = preds.flatten()[correct], y[correct]
        
    return images, targets, preds, X_correct, z_correct, preds_correct, labels_correct


def get_balanced_classes(num_per_class, X_correct, z_correct, preds_correct, labels_correct, num_classes, random_seed):
    """
    Call e.g. via X_correct, z_correct, preds_correct, labels_correct = \
        get_balanced_classes(10, X_correct, z_correct, preds_correct, labels_correct)
    """
    all_ind = []
    torch.manual_seed(random_seed)
    with torch.no_grad():
        for i in range(num_classes):
            i_ind_full = torch.where(labels_correct == i)[0]
            perm = torch.randperm(i_ind_full.size(0)).cpu()
            assert len(perm) >= num_per_class
            all_ind.extend(list(i_ind_full[perm[:num_per_class]].cpu()))
            all_ind.sort()
    return X_correct[all_ind], z_correct[all_ind], preds_correct[all_ind], labels_correct[all_ind]
        

def visualize_images(nr_images, X, y, preds, cifar10_labels):
    """
    Call e.g. via visualize_images(10, X, y, preds, cifar10_labels)
    """
    nr_x_axis = nr_images // 5
    # Visualize selected images
    fig, axes = plt.subplots(nr_x_axis, 5, figsize=(20,10))
    for i in range(nr_images):
      #  if correct[i]:
        data = np.transpose(X[i,:,:,:].cpu(), (1,2,0))
        x_ind = i%nr_x_axis
        y_ind = i%5
        axes[x_ind][y_ind].imshow(data)
        axes[x_ind][y_ind].set_title('Pred: {} \nLabel: {}'.format(cifar10_labels[preds[0][i]], cifar10_labels[y[i]]))
    print(y[:nr_images])
    print(preds[0][:nr_images])
    

def get_optimized_feature_z(z, W, b, label, num_classes, device, debug=False):
    """
    Call e.g. via z_adv, best_k, best_pred = get_optimized_feature_z(z, W, b, label, num_classes=num_classes)
    """
    
    # Get list on incorrect classes
    ks = [i for i in range(num_classes) if i != label]
    #print('List of incorrect classes for the selected image:', ks)

    # For each k: (Fix one k for now)
    out_label = W[label]@z+b[label]

    #out_k = W[k]@z+b[k]
    delta_ks = W[label]@z+b[label] - W@z+b # [out_label - W[k]@z+b[k] for k in ks]
    W_del = W[label] - W

    #### Optimize over all incorrect classes

    best_l2_norm_eps_k = np.inf
    best_pred = label

    # iterate over all incorrect classes
    for k in ks:
        assert k!= label
        delta_k = delta_ks[k].cpu().detach().numpy()
        W_del_k = W_del[k].cpu().detach().numpy()

        # Create a new model
        m = gp.Model("qp_"+str(k))
        m.Params.LogToConsole = 0

        # Add variable vector eps of dimensions 640x1
        eps = m.addMVar(640)

        # set objective function: min ||eps||_2^2
        m.setObjective(eps @ eps)

        # Add constraints --> THIS SHOULD BE A STRICT INEQUALITY: < 0 INSTEAD OF <= -0.1
        m.addConstr(W_del_k @ eps + delta_k <= -1e-8, name="c")

        m.optimize();

        eps_k = torch.from_numpy(eps.x)

        # Calculate z_adv (depending on class k):
        z_adv_k = z + eps_k.to(device)

        # First, get model output via applying last fully connected linear layer
        out = W@z_adv_k.float() + b
        out_org = W@z + b
        pred = torch.argmax(out)

        # Determine ||eps_k||_2
        l2_norm_eps_k = np.linalg.norm(eps_k)

        # Print summary
        if l2_norm_eps_k < best_l2_norm_eps_k and pred != label:
            best_l2_norm_eps_k = l2_norm_eps_k
            best_pred = pred
            best_k = k
            best_eps = eps_k

    if debug:
        print('\nBest L2-norm of eps: {}\nAchieved for k = {}\nPredicted class: {}\nCorrect class: {}'.format(best_l2_norm_eps_k,
                                                                                                          best_k,
                                                                                                          best_pred, label))
    z_dict = {'Best ||eps_k||_2': best_l2_norm_eps_k,
              'Best k': best_k,
              'Best pred': int(best_pred),
              'Label': int(label)}
    
    z_adv = z + best_eps.to(device)
    if debug:
        print('||z_adv - z||_2 = ||eps||_2 =', round(float(torch.norm(z-z_adv)),2))
    
    return z_adv, best_k, best_pred, z_dict


def model_out(x, z_adv, model2, device):
    start = time.time()
    model2.eval();
    criterion = nn.MSELoss()
    with torch.no_grad():
        x = x.to(device)
        outputs = model2(x)
        loss = criterion(outputs.flatten(), z_adv.detach().float())
    end = time.time()
    return loss


def get_projection(x, stepsize, grad_x, called_from):
    x_in = x.detach().clone()
    with torch.no_grad():
        x = torch.add(x, -stepsize * grad_x)
        x = torch.clip(x, 0, 1)  # ensure valid pixel range
    return x


def get_stepsize(x, grad_x, beta, sigma, z_adv, model2, device, debug=False):
    l = 0
    stepsize = beta**l
    x_step = get_projection(x, stepsize, grad_x, 'SS')
    with torch.no_grad():
        f_x_step = model_out(x_step, z_adv, model2, device)
        f_x = model_out(x, z_adv, model2, device)
        while float(f_x_step) > float(f_x - sigma * torch.sum(grad_x * (x-x_step))): #grad_x@(x-x_step):

            start = time.time()
            l += 1
            stepsize = beta**l
            x_step = get_projection(x, stepsize, grad_x, 'SS')
            #print('    Stepsize:', stepsize)
            f_x_step = model_out(x_step, z_adv, model2, device)
            f_x = model_out(x, z_adv, model2, device)
            end = time.time()
           # print('    Stepsize-Iteration {}: {} sec.'.format(l, round(end-start, 2)))
            if stepsize < 1e-6:
                if debug:
                    print('Break stepsize iteration after {} iterations.'.format(l))
                break
    return stepsize


def get_gradient(x, model2, z_adv, device):
    model2.eval();
    criterion = nn.L1Loss()#MSELoss()
    x = x.to(device)
    x.requires_grad = True
    outputs = model2(x)
    loss = criterion(outputs.flatten(), z_adv.detach().float())
    loss.backward()
    grad_x = x.grad.cpu()
    return loss, grad_x


def proj_grad_descent(X_org, beta, sigma, z_adv, model, model2, model_robust, label, device, num_steps = 100,
                     num_eval_steps = [], debug=False):
    
    losses = []
    L_infs = []
    x_opt_list = []
    i = 0
    x = X_org.clone()
    label = label.to(device)
    pred = label
    misclassified = False
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    
    steps = 0
    loss, grad_x = get_gradient(x, model2, z_adv, device)
    x_1 = get_projection(x, 1, grad_x, 'PGD')

    while torch.norm(x - x_1) > 1e-8  and steps < num_steps and pred == label:
        start = time.time()
        loss, grad_x = get_gradient(x, model2, z_adv, device)
        losses.append(float(loss))
        stepsize = get_stepsize(x, grad_x, beta, sigma, z_adv, model2, device, debug=debug)    
        x = get_projection(x, stepsize, grad_x, 'PDG')
        steps += 1
        with torch.no_grad():
            L_infs.append(torch.norm(X_org-x, p=np.inf))
            x_1 = get_projection(x, 1, grad_x, 'PGD')
            if debug:
                print('Linf-norm x-x(1):', torch.norm(x - x_1, p=np.inf))
        end = time.time()
        if debug:
            print('\nPGD-Iteration {}: {} sec.'.format(steps, round(end-start, 2)))
        
        X_prep = x.sub(mean[None, :, None, None]).div(std[None, :, None, None]).to(device)
        out = model(X_prep, feature = False)
        pred = torch.argmax(out)
        
        if pred != label:
            print('Image has been misclassified after {} iterations.'.format(steps))
            misclassified = True
            
        if steps % 5 == 0 and debug:
            get_iteration_report(z_adv, X_org, x, loss, label, steps, model, model_robust, device, final=False)
            
        if steps+1 in num_eval_steps or pred != label:
            step_dict = get_iteration_report(z_adv, X_org, x, loss, label, steps, model, model_robust, device, final=True)
            x_opt_list.append(step_dict)
            
        x_opt_df = pd.DataFrame(x_opt_list)
    return x, loss, x_opt_df, misclassified


def get_optimized_input_image(X_correct, z_correct, labels_correct, W, b, model, model2, model_robust, device,
                              num_eval_steps, num_classes, num_steps=100, beta=0.85, sigma=0.65, debug=False):
    """
    Call e.g. via x_opt_df = get_optimized_input_image(X_correct, z_correct, labels_correct, W, b, model2,
                              device, num_steps=100, beta=0.85, sigma=0.65)
    """
    z_opt_dict = {}
    x_opt_dfs = []
    misclassified_x = {}
    for image_nr in tqdm(range(len(labels_correct))):

        print('\n\nWorking on image number ', image_nr)

        # Fix one sample:
        z = z_correct[image_nr]
        label = labels_correct[image_nr]
        X_org = X_correct[image_nr].reshape((1,3,32,32)).detach().clone()

        z_adv, best_k, best_pred, z_dict = get_optimized_feature_z(z, W, b, label, num_classes, device)
        z_opt_dict[image_nr] = z_dict
        
        if best_pred != label:

            x, final_loss, x_opt_df, misclassified = proj_grad_descent(X_org, beta, sigma, z_adv, model, model2, model_robust, label,
                                              device, num_steps = num_steps, 
                                              num_eval_steps=num_eval_steps,
                                              debug=debug)

            # Display image
            display_image(X_org, x)

            x_opt_df['image_nr'] = image_nr
            x_opt_df['k_eps'] = best_k
            x_opt_df['eps_pred'] = int(best_pred)
            x_opt_dfs.append(x_opt_df)
            display(x_opt_df)
            if misclassified == True:
                misclassified_x[image_nr] = x
            
        else:
            print('Optimization for z failed.')
    
    z_opt_df = pd.DataFrame(z_opt_dict).T
    x_opt_dfs = pd.concat(x_opt_dfs, ignore_index=True)
    
    return z_opt_df, x_opt_dfs, misclassified_x


def display_image(X_org, x, i=0, save_path=False):
    plt.figure()
    fig, axes = plt.subplots(1, 2)#, figsize=(20,10))
    axes[0].imshow(np.transpose(X_org[i,:,:,:], (1,2,0)))
    axes[0].set_title('In')
    axes[1].imshow(np.transpose(x[i,:,:,:], (1,2,0)))
    axes[1].set_title('Out')
    if save_path != False:
        plt.savefig(save_path)


def get_iteration_report(z_adv, X_org, x, loss, label, step, model, model_robust, device, final=False, cifar_10_meta_path=CIFAR10_META_PATH, debug=False):
    if debug:
        print('----------------------------------------------------------------------')
        print('\n\nLoss:', loss, 
              '\nL_inf-norm:', torch.norm(X_org-x, p=np.inf))
    
    # get cifar labels:
    cifar10_labels = unpickle(cifar_10_meta_path)[b'label_names']
    
    # define mean and std for preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    
    x = x.to(device)
    X_org = X_org.to(device)
    # Get optimized input model prediction
    X_prep = x.sub(mean[None, :, None, None]).div(std[None, :, None, None])
    out, z = model(X_prep, feature = True)
    _, preds = out.topk(max((1,)), 1, True, True)
    if debug:
        print('\nRetrained preds x:',
              '\nPreds: {} (class {})'.format(cifar10_labels[preds[0,0]], preds[0,0]), 
              '\nLabel: {} (class {})'.format(cifar10_labels[label], label))
        print("L_inf(z'-m(x')):", torch.norm(z_adv-z, p=np.inf))
    if final:
        img_dict = {'Misclassified': bool(label != preds[0,0]), 
                    '||z-z_adv||_2': float(torch.norm(z_adv-z, p=2)),
                    '||z-z_adv||_inf': float(torch.norm(z_adv-z, p=np.inf)),
                    '||x-x_opt||_2': float(torch.norm(X_org-x, p=2)),
                    '||x-x_opt||_inf': float(torch.norm(X_org-x, p=np.inf)),
                    'preds': int(preds[0,0]),
                    'label': int(label),
                    'steps': step+1}
    
    # Get natural input model prediction
    X_prep = X_org.sub(mean[None, :, None, None]).div(std[None, :, None, None])
    out, z = model(X_prep, feature = True)
    _, preds = out.topk(max((1,)), 1, True, True)
    if debug:
        print('\nRetrained preds X_org:',
              '\nPreds: {} (class {})'.format(cifar10_labels[preds[0,0]], preds[0,0]), 
              '\nLabel: {} (class {})'.format(cifar10_labels[label], label))
        print("L_inf(z'-m(x)):", torch.norm(z_adv-z, p=np.inf))
    
    if final:
        # Get robust model prediction
        X_prep = x.sub(mean[None, :, None, None]).div(std[None, :, None, None])
        out, z = model_robust(X_prep, feature = True)
        _, preds = out.topk(max((1,)), 1, True, True)
        if debug:
            print('\nRobust preds x:',
                  '\nPreds: {} (class {})'.format(cifar10_labels[preds[0,0]], preds[0,0]), 
                  '\nLabel: {} (class {})'.format(cifar10_labels[label], label))
        return img_dict


def get_optimization_report_z(z_opt_df):

    print('#'*51+'\n#'+' '*15 +'Optimization Report' + ' '*15 + '#\n' +'#'*51)
    
    print('Number of misclassified feature vectors: {}/{}'. format(len(z_opt_df.loc[z_opt_df['Best pred'] != z_opt_df.Label]),
                                                                   len(z_opt_df)))
    
    print('Number of matches - k and pred: {}/{}'. format(len(z_opt_df.loc[z_opt_df['Best pred'] == z_opt_df['Best k']]),
                                                               len(z_opt_df)))
    
    print('\nStatistics - ||eps_k||_2:')
    display(z_opt_df['Best ||eps_k||_2'].agg(['min', 'mean', 'max', 'std']).reset_index().set_index('index').T)
    return len(z_opt_df.loc[z_opt_df['Best pred'] != z_opt_df.Label]), len(z_opt_df)


def get_optimization_report_x(x_opt_df):

    print('#'*51+'\n#'+' '*15 +'Optimization Report' + ' '*15 + '#\n' +'#'*51)
    print('\nNumber of misclassified images: {}/{}'.format(x_opt_df['Misclassified'].sum(),
                                                          len(x_opt_df['Misclassified'])))

    print('\nStatistics - Total:')
    display(x_opt_df[['||z-z_adv||_2', '||z-z_adv||_inf', '||x-x_opt||_2', '||x-x_opt||_inf']].agg(['min', 'mean', 'max', 'std']))

    print('\nStatistics - Misclassified:')
    display(x_opt_df.loc[x_opt_df.Misclassified == True][['||z-z_adv||_2', '||z-z_adv||_inf', '||x-x_opt||_2', '||x-x_opt||_inf']].agg(['min', 'mean', 'max', 'std']))

    print('\nStatistics - Correctly Classified:')
    display(x_opt_df.loc[x_opt_df.Misclassified == False][['||z-z_adv||_2', '||z-z_adv||_inf', '||x-x_opt||_2', '||x-x_opt||_inf']].agg(['min', 'mean', 'max', 'std']))


def run_hyperparameter_tuning(num_images, num_eval_steps, num_steps, sigmas, betas, model, model2, model_robust,
                              X_correct, z_correct, labels_correct, W, b, num_classes, device):
    
    hyperparameter_df = pd.DataFrame(columns = ['Misclassified', '||z-z_adv||_2', '||z-z_adv||_inf', 
                                                '||x-x_opt||_2', '||x-x_opt||_inf', 'preds', 'label', 'steps',
                                                'k_eps', 'eps_pred'])
    hyperparameter_dfs = []
    for i, (beta, sigma) in enumerate(itertools.product(betas,sigmas)):
        z_opt_dict, x_opt_df, misclassified_x = get_optimized_input_image(X_correct[:num_images], 
                                                        z_correct[:num_images], 
                                                        labels_correct[:num_images], 
                                                        W, b, 
                                                        model,
                                                        model2, 
                                                        model_robust,
                                                        device,
                                                        num_eval_steps = num_eval_steps,
                                                        num_classes=num_classes,
                                                        num_steps=num_steps, 
                                                        beta=beta, sigma=sigma, 
                                                        debug=False)
        x_opt_df['run_id'] = i
        x_opt_df['num_steps'] = num_steps
        x_opt_df['sigma'] = sigma
        x_opt_df['beta'] = beta
        x_opt_df['num_images'] = num_images
        hyperparameter_dfs.append(x_opt_df)
        
    hyperparameter_df = pd.concat(hyperparameter_dfs, ignore_index=True)

    return hyperparameter_df, misclassified_x


def eval_hyperparameter_df(hyperparameter_df):
    
    print('#'*51+'\n#'+' '*15 +'Optimization Report' + ' '*15 + '#\n' +'#'*51)
    
    print('Number of successful eps optimizations: {}/{}.'.format(len(hyperparameter_df.loc[hyperparameter_df\
                                                                    .label == hyperparameter_df['eps_pred']]), 
                                                                  len(hyperparameter_df)))
    
    print('\nNumber of misclassified images per step:')
    display(hyperparameter_df.groupby(['run_id', 'steps']).Misclassified.sum().reset_index())

    agg_cols = ['||z-z_adv||_2', '||z-z_adv||_inf', '||x-x_opt||_2', '||x-x_opt||_inf']

    print('\nStatistics - Total:')
    display(hyperparameter_df.groupby(['run_id', 'steps'])[agg_cols].agg(['min','mean','max','std']))

    print('\nStatistics - Misclassified:')
    display(hyperparameter_df.loc[hyperparameter_df.Misclassified == True]\
            .groupby(['run_id', 'steps'])[agg_cols].agg(('min','mean','max','std')))

    print('\nStatistics - Correctly Classified:')
    display(hyperparameter_df.loc[hyperparameter_df.Misclassified == False]\
            .groupby(['run_id', 'steps'])[agg_cols].agg(['min','mean','max','std']))
    
    print('\nMisclassified images - Details')
    display(hyperparameter_df.loc[hyperparameter_df.Misclassified == True])
                                                                  
    image_df = hyperparameter_df.drop_duplicates(['image_nr'])
    print('\nNumber of unique image_nrs with k_eps != eps_pred:', 
          len(image_df.loc[image_df.k_eps != image_df.eps_pred]))
    display(image_df.loc[image_df.k_eps != image_df.eps_pred])

    print('\nClass distribution:')    
    nr_k_df = image_df.groupby(['run_id', 'k_eps']).num_images.count().reset_index(name='nr_k')
    nr_label_df = image_df.groupby(['run_id', 'label']).num_images.count().reset_index(name = 'nr_label')
    nr_eps_pred_df = image_df.groupby(['run_id', 'eps_pred']).num_images.count().reset_index(name = 'nr_eps_pred')
    merge1 = nr_label_df.merge(nr_k_df, how = 'outer', right_on = ['run_id','k_eps'], left_on = ['run_id','label'])
    merge2 = merge1.merge(nr_eps_pred_df, how = 'left', left_on = ['run_id','label'], right_on = ['run_id','eps_pred'])
    merge2 = merge2.drop(columns = ['k_eps', 'eps_pred']).set_index('label').fillna(0)
    display(merge2)
    plt.figure()
    for col in ['nr_k', 'nr_label', 'nr_eps_pred']:
        plt.bar(merge2.index, merge2[col], alpha = 0.5, label = col)
        plt.title('Class histogram')
        plt.xlabel('Class')
        plt.ylabel('Samples per class')
        plt.legend()        
        
