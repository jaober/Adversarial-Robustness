# Adversarial Robustness

*While this project has been concluded, uploading the code files is currently a work in progress. Code needs to be restructured and simplified. A comprehensive documentation needs to be added on all levels.*

Research project on transferring adversarial robustness of deep learning models across domains.

Repository contains sample code for 
- preprocessing image data & defining data loaders 
- training a convolutional neural network on image data
- evaluating the adversarial robustness of a trained model
- retraining the last layers of a pretrained deep neural network
- hyperparameter tuning for retraining fully connected laywrs of a pretrained neural network
- median smoothing
- center smoothing
- obtaining theoretical lower bounds on the adversarial robustness of a retrained deep learning model

The code in this project has in part been inspired by and adapted from:  
- https://github.com/MadryLab/cifar10_challenge
- https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#wide_resnet50_2
- https://arxiv.org/pdf/1811.05381.pdf
- https://github.com/cemanil/LNets
- https://arxiv.org/abs/1804.00325
- https://github.com/bethgelab/foolbox/blob/master/examples/single_attack_pytorch_resnet18.py
- https://arxiv.org/pdf/2102.09701.pdf
- https://proceedings.neurips.cc/paper/2020/file/0dd1bc593a91620daecf7723d2235624-Paper.pdf.
