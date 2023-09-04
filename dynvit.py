import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset, build_transform
from engine import train_one_epoch, evaluate

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
from losses import ConvNextDistillDiffPruningLoss, DistillDiffPruningLoss_dynamic
from samplers import RASampler
from models.dyconvnext import ConvNeXt_Teacher, AdaConvNeXt
from models.dylvvit import LVViTDiffPruning, LVViT_Teacher
from models.dyvit import VisionTransformerDiffPruning, VisionTransformerTeacher
from models.dyswin import AdaSwinTransformer, SwinTransformer_Teacher
from calc_flops import calc_flops, throughput



def get_models(arch, num_classes):
    base_rate = 0.9
    criterion = torch.nn.CrossEntropyLoss()
    SPARSE_RATIO = [base_rate, base_rate - 0.2, base_rate - 0.4]

    if arch == 'swin_s':
        model = AdaSwinTransformer(
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                drop_rate=0.0,
                drop_path_rate=0,
                num_classes=num_classes,
                pruning_loc=[1,2,3], sparse_ratio=SPARSE_RATIO
            )
        # pretrained = torch.load('./pretrained/swin_tiny_patch4_window7_224.pth', map_location='cpu')
        teacher_model = SwinTransformer_Teacher(
                    embed_dim=96,
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    num_classes=num_classes,
                    window_size=7)
        
        criterion = ConvNextDistillDiffPruningLoss(
                    teacher_model, criterion, clf_weight=1.0, keep_ratio=SPARSE_RATIO, mse_token=True, ratio_weight=10.0, swin_token=True)
        
    elif arch == 'lvvit_s':
        PRUNING_LOC = [4,8,12] 
        KEEP_RATE = [SPARSE_RATIO[0], SPARSE_RATIO[0] ** 2, SPARSE_RATIO[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = LVViTDiffPruning(
                        patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
                    p_emb='4_2',skip_lam=2., return_dense=True,mix_token=True, num_classes=num_classes,
                    pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True
                )
        teacher_model = LVViT_Teacher(
                        patch_size=16, embed_dim=384, depth=16, num_heads=6, mlp_ratio=3.,
                    p_emb='4_2',skip_lam=2., num_classes=num_classes, return_dense=True, mix_token=True
                )
        
        criterion = DistillDiffPruningLoss_dynamic(
            teacher_model, criterion, clf_weight=1.0, keep_ratio=KEEP_RATE, mse_token=False, ratio_weight=2.0, distill_weight=0.5
        )
    elif arch == 'lvvit_m':
        PRUNING_LOC = [5,10,15] 
        KEEP_RATE = [SPARSE_RATIO[0], SPARSE_RATIO[0] ** 2, SPARSE_RATIO[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = LVViTDiffPruning(
            patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., num_classes=num_classes, return_dense=True,mix_token=True,
            pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True
        )
        teacher_model = LVViT_Teacher(
            patch_size=16, embed_dim=512, depth=20, num_heads=8, mlp_ratio=3.,
            p_emb='4_2',skip_lam=2., return_dense=True, mix_token=True, num_classes=num_classes
        )    

        criterion = DistillDiffPruningLoss_dynamic(
            teacher_model, criterion, clf_weight=1.0, keep_ratio=KEEP_RATE, mse_token=False, ratio_weight=2.0, distill_weight=0.5
        )

    elif arch == 'deit_s':
        PRUNING_LOC = [3, 6, 9]
        KEEP_RATE = [SPARSE_RATIO[0], SPARSE_RATIO[0] ** 2, SPARSE_RATIO[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, 
        pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, num_classes=num_classes
        )
        teacher_model = VisionTransformerTeacher(
            patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, num_classes=num_classes)  
          
        criterion = DistillDiffPruningLoss_dynamic(
            teacher_model, criterion, clf_weight=1.0, keep_ratio=KEEP_RATE, mse_token=True, ratio_weight=2.0, distill_weight=0.5
        )
    elif arch == 'deit_b':
        PRUNING_LOC = [3, 6, 9]
        KEEP_RATE = [SPARSE_RATIO[0], SPARSE_RATIO[0] ** 2, SPARSE_RATIO[0] ** 3]
        print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
        model = VisionTransformerDiffPruning(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, 
        pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True, drop_path_rate=0,
        )
        teacher_model = VisionTransformerTeacher(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True)
        
        criterion = DistillDiffPruningLoss_dynamic(
            teacher_model, criterion, clf_weight=1.0, keep_ratio=KEEP_RATE, mse_token=True, ratio_weight=2.0, distill_weight=0.5
        )
    else:
        raise Exception("No such model error")     
    return model, teacher_model, criterion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True)
    parser.add_argument("--base_rate", type=float, default='0.9')
    parser.add_argument('--drop_path', type=float, default=0, metavar='PCT',
                            help='Drop path rate (default: 0.0)')

    args = parser.parse_args()
    arch = args.arch



