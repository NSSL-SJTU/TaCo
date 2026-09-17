# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified for TaCo:
# Learning Flow Semantics for Encrypted Traffic Analysis:
# A Contrastive Pre-training Approach.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf
import numpy as np





# obtain sample
class TwoPNGCutOne(object):
    def __call__(self, x):
        x = np.array(x).copy()[:40, :]
        x = Image.fromarray(x.astype('uint8'))
        return x


# Flow-Level Augmentation 
class TwoPNGCutOneAug(object):
    def __init__(self, first = 1):
        self.first = first
    def __call__(self, x):
        x = np.array(x).copy()[self.first*8:40+self.first*8, :]
        x = Image.fromarray(x.astype('uint8'))
        return x



# Packet-Level Augmentation 
class PacketRetransmission(object):
    def __call__(self, x):
        retransmission_index = random.randint(0, 3)
        packets = np.array(x).copy().reshape(5, 8, 40)
        x = []
        for index in range(5):
            if index == retransmission_index:
                x.append(packets[index, :, :])
            x.append(packets[index, :, :])
            if len(x) == 5:
                break
        x = np.array(x).reshape(40, 40)
        x = Image.fromarray(x.astype('uint8'))
        return x




# Packet-Level Augmentation 
class PacketLoss(object):
    def __call__(self, x):
        loss_index = random.randint(0, 4)
        packets = np.array(x).copy().reshape(5, 8, 40)
        _, _, W = packets.shape
        no_packet = np.zeros([1, 8, W])
        x = np.delete(packets, loss_index, axis=0)
        x = np.append(x, no_packet, axis=0).reshape(-1, 40)
        x = Image.fromarray(x.astype('uint8'))
        return x

# Byte-Level Augmentation is in the model (vits.py)
