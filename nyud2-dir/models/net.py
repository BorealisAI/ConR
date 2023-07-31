# Copyright (c) 2023-present, Royal Bank of Canada.
# Copyright (c) 2022 Jiawei Ren
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Balanced MSE (https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html) implementation
# from https://github.com/jiawei-ren/BalancedMSE/tree/main/nyud2-dir by Jiawei Ren
####################################################################################
import torch
import torch.nn as nn
from models import modules

class model(nn.Module):
    def __init__(self, args, Encoder, num_features, block_channel):

        super(model, self).__init__()

        self.E = Encoder
        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R(args, block_channel)


    def forward(self, x, depth=None, epoch=None):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4,[x_decoder.size(2),x_decoder.size(3)])
        out,features = self.R(torch.cat((x_decoder, x_mff), 1), depth, epoch)

        return out,features
