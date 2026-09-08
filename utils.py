import wandb
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math

def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


def initialize(args):
    config_dict = namespace_to_dict(args)
    # check if already logged in
    name = f"{args.experiment_index:03d}-{args.instance_name}-{args.fold_name}" if args.fold_name is not None else f"{args.experiment_index:03d}-{args.instance_name}"
    if not wandb.api.api_key:
        wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=args.entity,
        project=args.project,
        name= name,
        config=config_dict,
        id=generate_run_id(name),
        reinit= True,
        resume="allow",)


def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)
