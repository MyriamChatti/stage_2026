# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 15:11:38 2026

@author: myria
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator_Base(nn.Module):
    def __init__(self):
        "Elle sert à gérer AdaIN dynamiquement dans le générateur"
        super(Generator_Base, self).__init__()

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        "parcourt le réseau, repère toutes les couches AdaIN, calcule combien de paramètres de style sont nécessaires au total"
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm":
                num_adain_params += 2 * m.num_features
        return num_adain_params

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        "ici le MLP produit un long vecteur, puis ce vecteur est découpé et chaque morceau est injecté dans une couche de adaIn"
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2 * m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

class AdaptiveInstanceNorm(nn.Module):
    "pour normaliser chaque image"
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        "défini le calcul du modèle"
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        self.weight=self.weight+1
        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

class MLP(nn.Module):
    "transforme le domaine avec AdaIn une image en paramètres continue (prend toutes les valeurs)"
    def __init__(self, input_dim, output_dim, dim, n_blk):
        super(MLP, self).__init__()
        self.model =[]
        self.model += [nn.Linear(input_dim,dim),nn.ReLU()]
        for i in range(n_blk - 2):
            self.model += [nn.Linear(dim, dim), nn.ReLU()]
        self.model += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

class ResBlk(nn.Module):
    "stabilité, meilleure propagation du gradient, standard dans les GAN modernes, AdaIN est dans le bloc, donc le style agit à chaque niveau."
    def __init__(self, planes, dimensions=2, padding_type='reflect'):
        super(ResBlk, self).__init__()
        assert dimensions in [2,3]
        Conv = nn.Conv2d if dimensions==2 else nn.Conv3d
        if padding_type == 'reflect':
            self.padm = nn.ReflectionPad2d(1) if dimensions==2 else nn.ReflectionPad3d(1)
            p=0
            #création de convolution 2D
        elif padding_type == 'replicate':
            self.padm = nn.ReplicationPad2d(1) if dimensions==2 else nn.ReplicationPad3d(1)
            p=0
        elif padding_type == 'zero':
            self.padm = nn.Identity()
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv1= Conv(planes,planes,3,padding=p)
        self.conv2 = Conv(planes,planes,3,padding=p)

        self.norm1 = AdaptiveInstanceNorm(num_features=planes)
        self.norm2 = AdaptiveInstanceNorm(num_features=planes)

        self.act = nn.ReLU()


    def forward(self, x):
        "c’est le passage des données d’entrée dans le modèle pour calculer la sortie"
        identity = x
        out = self.padm(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act(out)
        out = self.padm(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += identity
        return out #/ math.sqrt(2)