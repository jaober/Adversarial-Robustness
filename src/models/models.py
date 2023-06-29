"""
Implements model architectures used in this project.
Code adapted from:
 - https://github.com/MadryLab/cifar10_challenge
 - https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#wide_resnet50_2
 - Cem Anil, James Lucas, and Roger Grosse. Sorting out lipschitz function approximation, 2019.
   https://arxiv.org/pdf/1811.05381.pdf, https://github.com/cemanil/LNets
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from torch.nn.utils import weight_norm
import numpy as np
from torch.nn import Parameter


def conv3x3(in_planes, out_planes, stride, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class w_residual(nn.Module):

    def __init__(self, in_planes, out_planes, stride, activate_before_residual=False):
        super(w_residual, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.activate_before_residual = activate_before_residual
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.conv1 = conv3x3(in_planes=self.in_planes, out_planes=self.out_planes, stride=stride)

        self.bn1 = norm_layer(self.in_planes) # out_planes ??
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(self.out_planes, self.out_planes, stride=1)
        self.bn2 = norm_layer(self.out_planes)
        self.avg_pool = nn.AvgPool2d(stride, stride, padding = 0)

    def forward(self, x):
        if self.activate_before_residual:
            out = self.bn1(x)
            out = self.lrelu(out)
            identity = x
        else:
            identity = x
            out = self.bn1(x)
            out = self.lrelu(out)

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.in_planes != self.out_planes:
            identity = self.avg_pool(identity)
            identity = F.pad(identity, (0,0, 0,0, (self.out_planes-self.in_planes)//2, (self.out_planes-self.in_planes)//2, 0,0))

        out += identity

        return out


class WideResNet(nn.Module):

    def __init__(self, layers=32, widening_factor=10, num_classes=10, fc_layers=None, device='cuda'):
        super(WideResNet, self).__init__()

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1

        replace_stride_with_dilation = [False, False, False]

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        filters = [16, widening_factor*16, widening_factor*32, widening_factor*64]

        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = norm_layer(filters[3])
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.layer1 = self._make_layer(filters[0], filters[1], strides[0],
                     activate_before_residual[0])

        self.layer2 = self._make_layer(filters[1], filters[2], strides[1],
                     activate_before_residual[1])

        self.layer3 = self._make_layer(filters[2], filters[3], strides[2],
                     activate_before_residual[2])

        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))

        if fc_layers != None and fc_layers[0] == 'WN':
            WN_config = fc_layers[1]
            input_dim = WN_config['input_dim']
            layer_sizes = WN_config['layer_sizes'].copy()
            layer_sizes.insert(0, input_dim)
            if 'groupings' in WN_config.keys():
                groupings = WN_config['groupings'].copy()
                groupings.insert(0, -1)
                self.fc = self._get_sequential_WN_layers_gs(layer_sizes, groupings)
            else:
                self.fc = self._get_sequential_WN_layers_relu(layer_sizes)

        elif fc_layers != None and fc_layers[0] == 'LC':
            LC_config = fc_layers[1]
            input_dim = LC_config['input_dim']
            layer_sizes = LC_config['layer_sizes'].copy()
            layer_sizes.insert(0, input_dim)
            num_layers = len(layer_sizes)
            l_constant_per_layer = LC_config['l_constant'] ** (1.0 / (num_layers - 1))
            act_func = MaxMin
            groupings = LC_config['groupings'].copy()
            groupings.insert(0, -1)
            use_bias = LC_config['use_bias']

            self.fc = self._get_sequential_layers(layer_sizes, l_constant_per_layer, groupings, act_func, device,
                                                  use_bias=use_bias)
        else:
            self.fc = nn.Linear(filters[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_sequential_layers(self, layer_sizes, l_constant_per_layer, groupings, act_func, device, use_bias=True):
        layers = list()
        layers.append(StandardLinear(layer_sizes[0], layer_sizes[1], bias=use_bias))
        layers.append(Scale(l_constant_per_layer, device))

        for i in range(1, len(layer_sizes) - 1):
            downsampling_factor = (2.0 / groupings[i])
            layers.append(act_func(layer_sizes[i] // groupings[i]))

            layers.append(
                StandardLinear(int(downsampling_factor * layer_sizes[i]), layer_sizes[i + 1], bias=use_bias))
            layers.append(Scale(l_constant_per_layer, device))

        return nn.Sequential(*layers)


    def _get_sequential_WN_layers_gs(self, layer_sizes, groupings):
        layers = list()
        layers.append(weight_norm(nn.Linear(layer_sizes[0], layer_sizes[1]), name='weight', dim=0))

        for i in range(1, len(layer_sizes) - 1):
            downsampling_factor = (2.0 / groupings[i])
            layers.append(MaxMin(layer_sizes[i] // groupings[i]))
            layers.append(weight_norm(nn.Linear(int(downsampling_factor) * layer_sizes[i], layer_sizes[i+1]), name='weight', dim=0))

        return nn.Sequential(*layers)

    def _get_sequential_WN_layers_relu(self, layer_sizes):
        layers = list()
        layers.append(weight_norm(nn.Linear(layer_sizes[0], layer_sizes[1]), name='weight', dim=0))

        for i in range(1, len(layer_sizes) - 1):
            # Add the activation function.
            layers.append(nn.ReLU())
            layers.append(weight_norm(nn.Linear(layer_sizes[i], layer_sizes[i+1]), name='weight', dim=0))

        return nn.Sequential(*layers)

    def _make_layer(self, in_planes, out_planes, stride, activate_before_residual, blocks = 5): #, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        layers = []
        layers.append(w_residual(in_planes, out_planes, stride, activate_before_residual))

        for _ in range(1, blocks):
            layers.append(w_residual(out_planes, out_planes, 1, activate_before_residual))
        return nn.Sequential(*layers)

    def _forward_impl(self, x, feature):

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn3(x)
        x = self.relu(x)
        x = self.globalavgpool(x)
        z = torch.flatten(x, 1)
        x = self.fc(z)
        if feature:
            return (x, z)
        else:
            return x

    def forward(self, x, feature=False):
        return self._forward_impl(x, feature)

    def device(self):
        """
        Convenience function returning the device the model is located on.
        """
        return next(self.parameters()).device


def wideresnet(layers=32, widening_factor = 10, num_classes = 10, fc_layers = None, device = 'cuda'):
                    #arch, pretrained, progress, **kwargs):
    model = WideResNet(layers=layers, widening_factor = widening_factor, num_classes = num_classes,
                       fc_layers=fc_layers, device=device)

    return model

# load pretrained ResNet-18
class ClassificationCNN(nn.Module):
    def __init__(self):

        super(ClassificationCNN, self).__init__()
        resnet = torchvision.models.resnet18(pretrained=True)
        resnet.fc = nn.Linear(512, 10)
        self.model = resnet

    def forward(self, images):
        return self.model(images).squeeze()

def process_maxmin_size(x, num_units, axis=-1):
    size = list(x.size())
    num_channels = size[axis]

    if num_channels % num_units:
        raise ValueError('number of features({}) is not a '
                         'multiple of num_units({})'.format(num_channels, num_units))
    size[axis] = -1
    if axis == -1:
        size += [num_channels // num_units]
    else:
        size.insert(axis + 1, num_channels // num_units)
    return size


def maxout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.max(x.view(*size), sort_dim)[0]


def minout(x, num_units, axis=-1):
    size = process_maxmin_size(x, num_units, axis)
    sort_dim = axis if axis == -1 else axis + 1
    return torch.min(x.view(*size), sort_dim)[0]


class MaxMin(nn.Module):

    def __init__(self, num_units, axis=-1):
        super(MaxMin, self).__init__()
        self.num_units = num_units
        self.axis = axis

    def forward(self, x):
        maxes = maxout(x, self.num_units, self.axis)
        mins = minout(x, self.num_units, self.axis)
        maxmin = torch.cat((maxes, mins), dim=1)
        return maxmin

    def extra_repr(self):
        return 'num_units: {}'.format(self.num_units)


def project_network_weights(model, proj_config, device):
    for i, layer in enumerate(model.fc):
        if hasattr(model.fc[i], 'project_weights'):
            model.fc[i].project_weights(proj_config)

def get_linf_projection_threshold(weight, device):
    with torch.no_grad():
        if device == 'cpu':
            sorted_weights, _ = torch.abs(weight).sort(dim=1, descending=True)
            sorted_weights.float()
            partial_sums = torch.cumsum(sorted_weights, dim=1)
            indices = torch.arange(end=partial_sums.shape[1]).float()
            candidate_ks = (partial_sums < torch.tensor(1).float() +
                            (indices + torch.tensor(1).float()) * sorted_weights)
            candidate_ks = (candidate_ks.float() +
                            (1.0 / (2 * partial_sums.shape[1])) * (indices + torch.tensor(1).float()).float())
            _, ks = torch.max(candidate_ks.float(), dim=1)
            ks = ks.float()
            index_ks = torch.cat((torch.arange(end=weight.shape[0]).unsqueeze(-1).float(),
                                  ks.unsqueeze(1)), dim=1).long()

            thresholds = (partial_sums[index_ks[:, 0], index_ks[:, 1]] - torch.tensor(1).float()) / (
                    ks + torch.tensor(1).float())

        else:
            sorted_weights, _ = torch.abs(weight).sort(dim=1, descending=True)
            partial_sums = torch.cumsum(sorted_weights, dim=1)
            indices = torch.arange(end=partial_sums.shape[1]).float().cuda()
            candidate_ks = (partial_sums < torch.tensor(1).float().cuda() +
                            (indices + torch.tensor(1).float().cuda()) * sorted_weights)
            candidate_ks = (candidate_ks.float().cuda() +
                            (1.0 / (2 * partial_sums.shape[1])) * (indices +
                                                                   torch.tensor(1).float().cuda()).float())
            _, ks = torch.max(candidate_ks.float(), dim=1)
            ks = ks.float().cuda()
            index_ks = torch.cat((torch.arange(end=weight.shape[0]).unsqueeze(-1).float().cuda(),
                                  ks.unsqueeze(1)), dim=1).long()

            thresholds = (partial_sums[index_ks[:, 0], index_ks[:, 1]] - torch.tensor(1).float().cuda()) / (
                    ks + torch.tensor(1).float().cuda())
    return thresholds


class StandardLinear(nn.Module):
    r"""Applies a linear transformation to the incoming distrib: :math:`y = Ax + b`"""

    def __init__(self, in_features, out_features, bias=True):
        super(StandardLinear, self).__init__()
        self._set_network_parameters(in_features, out_features, bias) #MT: Set this to true later --> Cuda true/false

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def _set_network_parameters(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        # Set weights and biases.
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        nn.init.orthogonal_(self.weight, gain=stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def project_weights(self, proj_config):
        with torch.no_grad():
            thresholds = get_linf_projection_threshold(self.weight, False)
            signs = torch.sign(self.weight)
            signs[signs == 0] = 1
            projected_weights = signs * torch.clamp(torch.abs(self.weight) - thresholds.unsqueeze(-1),
                                                    min=torch.tensor(0).float())
            self.weight.data.copy_(projected_weights)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

class Scale(nn.Module):
    r"""Scales the input vector by a given scalar."""
    def __init__(self, factor, device): #cuda=False):
        super(Scale, self).__init__()
        self.factor = torch.Tensor([factor]).to(device)


    def forward(self, input):
        if self.factor == 1:
            return input
        else:
            return self.factor * input

    def extra_repr(self):
        return 'factor={}'.format(self.factor)


class FCNet(nn.Module):
    def __init__(self, layers, input_dim, l_constant, bias=True, dropout=False):
        super(FCNet, self).__init__()

        self.input_dim = input_dim
        self.layer_sizes = layers.copy()
        self.layer_sizes.insert(0, self.input_dim)
        self.l_constant = l_constant
        self.num_layers = len(self.layer_sizes)

        self.act_func = MaxMin

        self.groupings = [2,2,2,1]
        self.groupings.insert(0, -1)
        self.use_bias = bias
        self.linear = StandardLinear
        layers = self._get_sequential_layers(l_constant_per_layer=self.l_constant ** (1.0 / (self.num_layers - 1)),
                                             dropout=dropout)
        self.model = nn.Sequential(*layers)

    def __len__(self):
        return len(self.model)

    def __getitem__(self, idx):
        return self.model[idx]

    def forward(self, x):
        x = x.view(-1, self.input_dim)

        return self.model(x)

    def _get_sequential_layers(self, l_constant_per_layer, dropout=False):
        layers = list()
        if dropout:
            layers.append(nn.Dropout(0.2))
        layers.append(self.linear(self.layer_sizes[0], self.layer_sizes[1], bias=self.use_bias))
        layers.append(Scale(l_constant_per_layer, cuda=False))

        for i in range(1, len(self.layer_sizes) - 1):
            downsampling_factor = (2.0 / self.groupings[i])
            layers.append(self.act_func(self.layer_sizes[i] // self.groupings[i]))

            if dropout:
                layers.append(nn.Dropout(0.5))

            layers.append(
                self.linear(int(downsampling_factor * self.layer_sizes[i]), self.layer_sizes[i + 1], bias=self.use_bias))
            layers.append(Scale(l_constant_per_layer, cuda=False))

        return layers

    def get_activations(self, x):
        activations = []
        x = x.view(-1, self.input_dim)
        for m in self.model:
            x = m(x)
            if not isinstance(m, StandardLinear) and not isinstance(m, Scale) and not isinstance(m, nn.Dropout):
                activations.append(x.detach().clone())
        return activations


class WideResNet_gurobi(nn.Module):

    def __init__(self, layers=32, widening_factor=10, num_classes=10, fc_layers=None, device='cuda'):
        super(WideResNet_gurobi, self).__init__()

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1

        replace_stride_with_dilation = [False, False, False]

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        filters = [16, widening_factor*16, widening_factor*32, widening_factor*64]

        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = norm_layer(filters[3])
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.layer1 = self._make_layer(filters[0], filters[1], strides[0],
                     activate_before_residual[0])

        self.layer2 = self._make_layer(filters[1], filters[2], strides[1],
                     activate_before_residual[1])

        self.layer3 = self._make_layer(filters[2], filters[3], strides[2],
                     activate_before_residual[2])

        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))

        if fc_layers != None and fc_layers[0] == 'WN':
            WN_config = fc_layers[1]
            input_dim = WN_config['input_dim']
            layer_sizes = WN_config['layer_sizes'].copy()
            layer_sizes.insert(0, input_dim)

            self.fc = self._get_sequential_WN_layers(layer_sizes)

        elif fc_layers != None and fc_layers[0] == 'LC':
            LC_config = fc_layers[1]
            input_dim = LC_config['input_dim']
            layer_sizes = LC_config['layer_sizes'].copy()
            layer_sizes.insert(0, input_dim)
            num_layers = len(layer_sizes)
            l_constant_per_layer = LC_config['l_constant'] ** (1.0 / (num_layers - 1))
            act_func = MaxMin
            groupings = LC_config['groupings'].copy()
            groupings.insert(0, -1)
            use_bias = LC_config['use_bias']

            self.fc = self._get_sequential_layers(layer_sizes, l_constant_per_layer, groupings, act_func, device,
                                                  use_bias=use_bias)
        else:
            self.fc = nn.Linear(filters[-1], num_classes)


        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_sequential_layers(self, layer_sizes, l_constant_per_layer, groupings, act_func, device, use_bias=True):
        layers = list()
        layers.append(StandardLinear(layer_sizes[0], layer_sizes[1], bias=use_bias))
        layers.append(Scale(l_constant_per_layer, device))

        for i in range(1, len(layer_sizes) - 1):
            downsampling_factor = (2.0 / groupings[i])
            layers.append(act_func(layer_sizes[i] // groupings[i]))
            layers.append(
                StandardLinear(int(downsampling_factor * layer_sizes[i]), layer_sizes[i + 1], bias=use_bias))
            layers.append(Scale(l_constant_per_layer, device))

        return nn.Sequential(*layers)

    def _get_sequential_WN_layers(self, layer_sizes):
        layers = list()
        layers.append(weight_norm(nn.Linear(layer_sizes[0], layer_sizes[1]), name='weight', dim=0))

        for i in range(1, len(layer_sizes) - 1):
            layers.append(nn.ReLU())
            layers.append(weight_norm(nn.Linear(layer_sizes[i], layer_sizes[i+1]), name='weight', dim=0))

        return nn.Sequential(*layers)

    def _make_layer(self, in_planes, out_planes, stride, activate_before_residual, blocks = 5): #, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None

        layers = []
        layers.append(w_residual(in_planes, out_planes, stride, activate_before_residual))
        for _ in range(1, blocks):
            layers.append(w_residual(out_planes, out_planes, 1, activate_before_residual))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, z, feature):

        x = x.sub(self.mean[None, :, None, None]).div(self.std[None, :, None, None])
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.globalavgpool(x)
        x = torch.flatten(x, 1)
        if z != None:
            print('x', x.shape, 'z', z.shape)
            x = torch.norm(x-z, dim=1)
        return x

    def forward(self, x, feature=False):
        if type(x) == list:
            return self._forward_impl(x[0], x[1], feature)
        else:
            z = None
            return self._forward_impl(x, z, feature)

    def device(self):
        """
        Convenience function returning the device the model is located on.
        """
        return next(self.parameters()).device


class WideResNet(nn.Module):

    def __init__(self, layers=32, widening_factor=10, num_classes=10, fc_layers=None, device='cuda'):
        super(WideResNet, self).__init__()

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.dilation = 1

        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)

        replace_stride_with_dilation = [False, False, False]
        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]

        filters = [16, widening_factor*16, widening_factor*32, widening_factor*64]


        # Passt padding here ??
        self.conv1 = nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn3 = norm_layer(filters[3])
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        self.layer1 = self._make_layer(filters[0], filters[1], strides[0],
                     activate_before_residual[0])

        self.layer2 = self._make_layer(filters[1], filters[2], strides[1],
                     activate_before_residual[1])

        self.layer3 = self._make_layer(filters[2], filters[3], strides[2],
                     activate_before_residual[2])

        self.globalavgpool = nn.AdaptiveAvgPool2d((1, 1))

        if fc_layers != None and fc_layers[0] == 'WN':
            WN_config = fc_layers[1]
            input_dim = WN_config['input_dim']
            layer_sizes = WN_config['layer_sizes'].copy()
            layer_sizes.insert(0, input_dim)
            if 'groupings' in WN_config.keys():
                groupings = WN_config['groupings'].copy()
                groupings.insert(0, -1)
                self.fc = self._get_sequential_WN_layers_gs(layer_sizes, groupings)
            else:
                self.fc = self._get_sequential_WN_layers_relu(layer_sizes)

        elif fc_layers != None and fc_layers[0] == 'LC':
            LC_config = fc_layers[1]
            input_dim = LC_config['input_dim']
            layer_sizes = LC_config['layer_sizes'].copy()
            layer_sizes.insert(0, input_dim)
            num_layers = len(layer_sizes)
            l_constant_per_layer = LC_config['l_constant'] ** (1.0 / (num_layers - 1))
            act_func = MaxMin
            groupings = LC_config['groupings'].copy()
            groupings.insert(0, -1)
            use_bias = LC_config['use_bias']

            self.fc = self._get_sequential_layers(layer_sizes, l_constant_per_layer, groupings, act_func, device,
                                                  use_bias=use_bias)
        else:
            self.fc = nn.Linear(filters[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _get_sequential_layers(self, layer_sizes, l_constant_per_layer, groupings, act_func, device, use_bias=True):
        layers = list()
        layers.append(StandardLinear(layer_sizes[0], layer_sizes[1], bias=use_bias))
        layers.append(Scale(l_constant_per_layer, device))

        for i in range(1, len(layer_sizes) - 1):
            downsampling_factor = (2.0 / groupings[i])
            layers.append(act_func(layer_sizes[i] // groupings[i]))
            layers.append(
                StandardLinear(int(downsampling_factor * layer_sizes[i]), layer_sizes[i + 1], bias=use_bias))
            layers.append(Scale(l_constant_per_layer, device))

        return nn.Sequential(*layers)


    def _get_sequential_WN_layers_gs(self, layer_sizes, groupings):
        layers = list()
        layers.append(weight_norm(nn.Linear(layer_sizes[0], layer_sizes[1]), name='weight', dim=0))

        for i in range(1, len(layer_sizes) - 1):
            downsampling_factor = (2.0 / groupings[i])
            layers.append(MaxMin(layer_sizes[i] // groupings[i]))
            layers.append(weight_norm(nn.Linear(int(downsampling_factor) * layer_sizes[i], layer_sizes[i+1]), name='weight', dim=0))

        return nn.Sequential(*layers)

    def _get_sequential_WN_layers_relu(self, layer_sizes):
        layers = list()
        layers.append(weight_norm(nn.Linear(layer_sizes[0], layer_sizes[1]), name='weight', dim=0))

        for i in range(1, len(layer_sizes) - 1):
            layers.append(nn.ReLU())
            layers.append(weight_norm(nn.Linear(layer_sizes[i], layer_sizes[i+1]), name='weight', dim=0))
        return nn.Sequential(*layers)

    def _make_layer(self, in_planes, out_planes, stride, activate_before_residual, blocks = 5): #, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        layers = []
        layers.append(w_residual(in_planes, out_planes, stride, activate_before_residual))

        for _ in range(1, blocks):
            layers.append(w_residual(out_planes, out_planes, 1, activate_before_residual))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, feature):

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.globalavgpool(x)
        z = torch.flatten(x, 1)
        x = self.fc(z)
        if feature:
            return (x, z)
        else:
            return x

    def forward(self, x, feature=False):
        return self._forward_impl(x, feature)

    def device(self):
        """
        Convenience function returning the device the model is located on.
        """
        return next(self.parameters()).device


def wideresnet(layers=32, widening_factor = 10, num_classes = 10, fc_layers = None, device = 'cuda'):
    model = WideResNet(layers=layers, widening_factor = widening_factor, num_classes = num_classes,
                       fc_layers=fc_layers, device=device)

    return model
