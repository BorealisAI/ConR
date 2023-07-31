# Copyright (c) 2023-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

def ConR(feature,depth,output,weights,w=0.2,t=0.07,e=0.2):

    
    
    k = feature.reshape([feature.shape[0],-1])
    q = feature.reshape([feature.shape[0],-1])

    
    depth = depth.reshape(depth.shape[0],-1)
    l_k = torch.mean(depth,dim=1).unsqueeze(-1)
    l_q = torch.mean(depth,dim=1).unsqueeze(-1)

    output = output.reshape(output.shape[0],-1)
    p_k = torch.mean(output,dim=1).unsqueeze(-1)
    p_q = torch.mean(output,dim=1).unsqueeze(-1)
    
    
    
    
    l_dist = torch.abs(l_q -l_k.T)
    p_dist = torch.abs(p_q -p_k.T)

    


    q = torch.nn.functional.normalize(q, dim=1)
    k = torch.nn.functional.normalize(k, dim=1)
  
    Temp = 0.07
   
    # dot product of anchor with positives. Positives are keys with similar label
    pos_i = l_dist.le(w)
    neg_i = ((~ (l_dist.le(w)))*(p_dist.le(w)))

    for i in range(pos_i.shape[0]):
        pos_i[i][i] = 0
    
    prod = torch.einsum("nc,kc->nk", [q, k])/t
    pos = prod * pos_i
    neg = prod * neg_i


    
    #  Pushing weight 
    weights = torch.mean(weights.reshape(weights.shape[0],-1),dim=1).unsqueeze(-1)
    pushing_w = l_dist*weights*e
    

    # Sum exp of negative dot products
    neg_exp_dot=(pushing_w*(torch.exp(neg))*(neg_i)).sum(1)

    # For each query sample, if there is no negative pair, zero-out the loss.
    no_neg_flag = (neg_i).sum(1).bool()

    # Loss = sum over all samples in the batch (sum over (positive dot product/(negative dot product+positive dot product)))
    denom=l_dist.le(w).sum(1)

    loss = ((-torch.log(torch.div(torch.exp(pos),(torch.exp(pos).sum(1) + neg_exp_dot).unsqueeze(-1)))*(pos_i)).sum(1)/denom)
    
    
    
    
    loss = ((loss*no_neg_flag).unsqueeze(-1)).mean()
    
    

    return loss