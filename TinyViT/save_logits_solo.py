# --------------------------------------------------------
# TinyViT Save Teacher Logits
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# Save teacher logits
# --------------------------------------------------------

import os
import time
import random
import argparse
import datetime
from collections import defaultdict
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.utils import accuracy
from my_meter import AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from logger import create_logger
from utils import load_checkpoint, NativeScalerWithGradNormCount, add_common_args

from models.remap_layer import RemapLayer
remap_layer_22kto1k = RemapLayer('./imagenet_1kto22k.txt')

JUST_SHOW_CONIFGS = False

def parse_option():
    parser = argparse.ArgumentParser(
        'TinyViT saving sparse logits script', add_help=False)
    add_common_args(parser)
    parser.add_argument('--check-saved-logits',
                        action='store_true', help='Check saved logits')
    parser.add_argument('--skip-eval',
                        action='store_true', help='Skip evaluation')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main():
    pass


if __name__ == '__main__':
    args, config = parse_option()

    config.defrost()
    assert len(
        config.DISTILL.TEACHER_LOGITS_PATH) > 0, "Please fill in the config DISTILL.TEACHER_LOGITS_PATH"
    config.DISTILL.ENABLED = True
    if not args.check_saved_logits:
        config.DISTILL.SAVE_TEACHER_LOGITS = True
    config.freeze()

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    # torch.cuda.set_device(config.LOCAL_RANK)
    # torch.distributed.init_process_group(
    #     backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    # torch.distributed.barrier()

    # The seed changes with config, rank, world_size and epoch
    # seed = config.SEED + dist.get_rank() + config.TRAIN.START_EPOCH * \
    #     dist.get_world_size()
    seed = config.SEED + config.TRAIN.START_EPOCH
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    # if dist.get_rank() == 0:
    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    if not JUST_SHOW_CONIFGS:
        main()
    else:
        print('========== END ============')