import os
import glob
import torch
import torchvision.utils as vutils
from network import VoiceEmbedNet, Generator
import webrtcvad

from mfcc import MFCC
from network import get_network
from utils import npy2face, getNpy, getMultipleNpy
import numpy as np

# initialization

NETWORKS_PARAMETERS = {
    # VOICE EMBEDDING NETWORK (e)
    'e': {
        'network': VoiceEmbedNet,
        'input_channel': 64,
        'channels': [256, 384, 576, 864],
        'output_channel': 64, # the embedding dimension
        'model_path': 'pretrained_models/voice_embedding.pth',
    },
    # GENERATOR (g)
    'g': {
        'network': Generator,
        'input_channel': 64,
        'channels': [1024, 512, 256, 128, 64], # channels for deconvolutional layers
        'output_channel': 3, # images with RGB channels
        'model_path': 'pretrained_models/generator_crop_50.pth',
    },
    'GPU': True}

e_net, _ = get_network('e', NETWORKS_PARAMETERS, train=False)
g_net, _ = get_network('g', NETWORKS_PARAMETERS, train=False)

ids_list = [10190, 10167, 10182, 10306, 10498]

old_list = [10090, 10086, 10116, 10303, 10203, 10242, 10096, 10088]

# for ids in ids_list:
#     arr = getNpy(ids)
#     print(arr.shape)
#     for t in [100, 200, 300, 500, 1000]:
#         face_image = npy2face(e_net, g_net, arr[:t, :], NETWORKS_PARAMETERS['GPU'])
#         vutils.save_image(face_image.detach().clamp(-1,1),
#                             'images/' + str(ids) + '_' + str(t) +'.png', normalize=True)

for ids in ids_list:
    arr = getNpy(ids)
    face_image = npy2face(e_net, g_net, arr, GPU=True)
    vutils.save_image(face_image.detach().clamp(-1,1),
                        'temp/' + str(ids) +'.png', normalize=True)
# for ids in ids_list:
#     arrs = getMultipleNpy(ids)
#     for i in range(5):
#         face_image = npy2face(e_net, g_net, arrs[i], NETWORKS_PARAMETERS['GPU'])
#         vutils.save_image(face_image.detach().clamp(-1,1),
#                             'images/' + str(ids) + '_1' + str(i) +'.png', normalize=True)

