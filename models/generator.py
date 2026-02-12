# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 15:53:57 2026

@author: myria
"""

from .networks import *#importe tous les blocs

class GeneratorDecoder(Generator_Base):
    def __init__(self, image_nc, ngf=64, num_domain=2, dimensions=2):
        super(GeneratorDecoder, self).__init__()
        
        # Choix automatique Conv2D / Conv3D selon le type de données
        Conv = nn.Conv2d if dimensions == 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if dimensions == 2 else nn.ConvTranspose3d
        pad = nn.ReflectionPad2d if dimensions == 2 else nn.ReflectionPad3d
        main = []
        #4 blocs résiduels
        # Ils traitent la représentation de contenu sans changer la résolution
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ConvTranspose(ngf * 4, ngf * 2, 3, 2, 1, 1), AdaptiveInstanceNorm(ngf * 2), nn.ReLU()]
        # Upsampling 1 : augmente la résolution (x2)
        #AdaIN : injection du style du domaine cible
        main += [ConvTranspose(ngf * 2, ngf, 3, 2, 1, 1), AdaptiveInstanceNorm(ngf), nn.ReLU()]
        #Upsampling 2 : retour vers la résolution d’origine
        main += [pad(3), Conv(ngf, image_nc, 7)]
        #Couche finale : reconstruction image
        
        self.main = nn.Sequential(*main)
        # Assemble toutes les couches
        
        
        
        #MLP de style
        # Entrée : label de domaine (one-hot)
        # Sortie : tous les paramètres AdaIN nécessaires au décodeur
        self.mlp = MLP(num_domain, self.get_num_adain_params(self.main), 64, 3)

    def forward(self, x, D_c,return_logits_only=False):
        # Création d’un vecteur one-hot pour le domaine
        ones = torch.sparse.torch.eye(2)
        # Le MLP génère tous les paramètres AdaIN
        adain_params = self.mlp(ones.index_select(0, D_c))
        # Injection des paramètres AdaIN dans les couches
        self.assign_adain_params(adain_params, self.main)
        
        # Décodage
        x = self.main(x)
        
        #Option utilisée pendant l’entraînement adversarial
        if return_logits_only:return x
        return x.tanh()

class GeneratorEncoder(Generator_Base):
    def __init__(self, image_nc, ngf=64, num_domain=2, dimensions=2):
        super(GeneratorEncoder, self).__init__()
        Conv = nn.Conv2d if dimensions == 2 else nn.Conv3d
        ConvTranspose = nn.ConvTranspose2d if dimensions == 2 else nn.ConvTranspose3d
        pad = nn.ReflectionPad2d if dimensions == 2 else nn.ReflectionPad3d

        main = []
        main += [pad(3), Conv(image_nc, ngf, 7), AdaptiveInstanceNorm(ngf), nn.ReLU()]
        main += [Conv(ngf, ngf * 2, 3, 2, 1), AdaptiveInstanceNorm(ngf * 2), nn.ReLU()]
        main += [Conv(ngf * 2, ngf * 4, 3, 2, 1), AdaptiveInstanceNorm(ngf * 4), nn.ReLU()]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        main += [ResBlk(ngf * 4, dimensions=dimensions)]
        self.main = nn.Sequential(*main)
        self.mlp = MLP(num_domain, self.get_num_adain_params(self.main), 64, 3)

    def forward(self, x, E_c):
        ones = torch.sparse.torch.eye(2)
        adain_params = self.mlp(ones.index_select(0, E_c))
        self.assign_adain_params(adain_params, self.main)
        x = self.main(x)
        return x