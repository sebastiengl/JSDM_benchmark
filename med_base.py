# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""


import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import logging
import os

from models import MarginalModel, AutoRegModel
from samplers import SSESampler, MarginalSampler, AutoRegSampler
from torch.utils.data.distributed import DistributedSampler
import utils
import pandas as p
from dataset import SpeciesVocab, TabularDataset
from tqdm import tqdm
from config import as_namespace, load_config

#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TO_DO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha= 1 - decay)

    for ema_buf, buf in zip(ema_model.buffers(), model.buffers()):
        ema_buf.copy_(buf)
    
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def init_main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    if "RANK" not in os.environ:
        # Single-GPU mode: set default env vars
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
    
    dist.init_process_group("nccl")

    assert args.batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

def create_dir(args):
    os.makedirs(args.results_dir, exist_ok=True)
    existing_experiments = glob(f'{args.results_dir}/*')
    experiment_index = 1 if not existing_experiments else max(
        int(path.rsplit("/", 1)[-1][:3]) for path in existing_experiments) + 1
    experiment_dir = f'{args.results_dir}/{experiment_index:03d}-Experiment'
    os.makedirs(experiment_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    args.logger = logger
    args.experiment_dir = experiment_dir
    args.experiment_index = experiment_index

def prepare_data(args):
    species_df = p.read_csv(args.train_path)
    test_df = None if args.test_path is None else p.read_csv(args.test_path)
    vocab = SpeciesVocab(train_species=species_df, test_species= species_df if test_df is None else test_df)
    species_ds = globals()[args.dataset_class](species=species_df, vocab=vocab, subset="train", x_dir=args.metadata_path, **args.data_args)
    test_ds = None if test_df is None else globals()[args.dataset_class](species=test_df, vocab=vocab, subset="test", x_dir=args.metadata_path, **args.data_args)
    fold_map = None
    if args.test_path is None :
        fold_df = p.read_csv(args.kfold_path)
        if args.index_column not in fold_df or args.index_fold not in fold_df:
            raise ValueError(f"{args.kfold_path} must contain {args.index_column!r} and {args.index_fold!r} columns")
        fold_map = dict(zip(fold_df[args.index_column], fold_df[args.index_fold]))
    return vocab, species_ds, test_ds, fold_map, {}


def build_loaders(args, data):
    _, species_ds, test_ds, fold_map, loader_cache = data
    if args.fold_name in loader_cache:
        return loader_cache[args.fold_name]
    if args.fold_name is not None:
        train_indices = [i for i, survey_id in enumerate(species_ds.surveys) if fold_map.get(survey_id) != args.fold_name]
        test_indices = [i for i, survey_id in enumerate(species_ds.surveys) if fold_map.get(survey_id) == args.fold_name]
        test_ds = torch.utils.data.Subset(species_ds, test_indices)
        trainval_ds = torch.utils.data.Subset(species_ds, train_indices)
    else :
        trainval_ds = species_ds
    train_ds, val_ds = torch.utils.data.random_split(trainval_ds, [0.8, 0.2])
    local_batch_size = args.batch_size // dist.get_world_size()
    train_sampler = DistributedSampler(train_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, seed=args.seed)
    val_sampler = DistributedSampler(val_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, seed=args.seed)
    test_sampler = DistributedSampler(test_ds, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, seed=args.seed)
    loaders = (
        DataLoader(train_ds, batch_size=local_batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler),
        DataLoader(val_ds, batch_size=local_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True, sampler=val_sampler),
        DataLoader(test_ds, batch_size=local_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False, sampler=test_sampler)
    )
    loader_cache[args.fold_name] = loaders
    return loaders


def main(args, data):
    """
    Trains a new model.
    """

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0

    # Setup an experiment folder:
    if rank == 0:
        logger = args.logger
        experiment_dir = args.experiment_dir
        instance_name = args.instance_name.replace("/", "-")
        fold_name = fold_list[args.fold]
        instance_dir = f"{experiment_dir}/{instance_name}/fold_{fold_name}" if fold_name is not None else f"{experiment_dir}/{instance_name}"
        checkpoint_dir = f"{instance_dir}/checkpoints"  # Stores this model's checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"Model directory created at {instance_dir}")
        if args.wandb:
            utils.initialize(args)
    else:
        logger = create_logger(None)



    vocab, species_ds, _, _, _ = data
    train_loader, val_loader, test_loader = build_loaders(args, data)


    ##################################################################################

    resume_state = None
    if args.ckpt is not None:
        assert os.path.isfile(args.ckpt), f'Could not find SiT checkpoint at {args.ckpt}'
        resume_state = torch.load(args.ckpt, map_location="cpu")
        args.model = resume_state.get("model_type")
        args.model_args = resume_state.get("model_args")

    model_kwargs = dict(args.model_args)
    model_kwargs["vocab"] = vocab
    model_kwargs.setdefault("env_emb_class", "TabularEmbedder")
    model_kwargs.setdefault("input_size", len(args.data_args["var_cols"]))
    if args.model == "AutoRegModel":
        model_kwargs.setdefault("max_length", species_ds.max_length)
    model = globals()[args.model](**model_kwargs).to(device)


    if args.emb_ckpt is not None:
        ckpt_path = args.emb_ckpt
        assert os.path.isfile(ckpt_path), f'Could not find env_embedder checkpoint at {ckpt_path}'
        state_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        state_dict["null_embedding"] = model.env_embedder.null_embedding.data
        model.env_embedder.load_state_dict(state_dict)
        logger.info(f"Loaded env_embedder checkpoint from {ckpt_path}")

    if args.emb_freeze:
        requires_grad(model.env_embedder, False)
        logger.info("env_embedder parameters are frozen.")

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training


    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)


    # Load from checkpoint if specified:
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        ema.load_state_dict(resume_state["ema"])
        opt.load_state_dict(resume_state["opt"])
        logger.info(f"Loaded SiT checkpoint from {args.ckpt}")

    model = model.to(device)
    use_ddp = dist.get_world_size() > 1
    if use_ddp:
        model = DDP(model, device_ids=[device], output_device=device)
    model_for_train = model.module if use_ddp else model
    compute_loss = model.module.loss_func



    sampler_kwargs = dict(args.sampler_args)
    sampler_kwargs["vocab"] = vocab
    sampler_kwargs["model"] = ema
    if args.sampler == "AutoRegSampler":
        sampler_kwargs["max_length"] = species_ds.max_length
    sample_fn = globals()[args.sampler](**sampler_kwargs).sample

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")


    #########################################################################

    # Prepare models for training:
    update_ema(ema, model_for_train, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()


    logger.info(f"Training {args.model} Fold {args.fold}...")

    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch+1}...")
        train_loader.sampler.set_epoch(epoch)
        for survey, species, tokens, y in tqdm(train_loader):
            tokens = tokens.to(device).long()
            species = species.to(device).float()
            for i in range(len(y)):
                y[i] = y[i].to(device)

            # Forward
            logits = model(y, tokens)
            loss = compute_loss(logits, species, tokens)
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model_for_train, decay = 0.999)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    utils.log(
                        { "train/loss": avg_loss, "train steps/sec": steps_per_sec },
                        step=train_steps)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save AutoReg checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model_for_train.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "model_type": args.model,
                        "model_args": args.model_args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if train_steps % args.sample_every == 0:
                metrics = {}
                with torch.no_grad():
                    train_f1 = torch.tensor(0., device=device)
                    logger.info("Generating EMA samples...")


                    samples = sample_fn(y).squeeze(1)

                    for i in range(samples.shape[0] ):
                        pred = samples[i]
                        tp = (pred * species[i]).sum()
                        if tp > 0 :
                            train_f1 += 2 * tp / (species[i].sum() + pred.sum())

                    dist.all_reduce(train_f1, op=dist.ReduceOp.SUM)
                    train_f1 /= args.batch_size
                    train_f1 = train_f1.item()
                    metrics["train/f1"]= train_f1

    
                    #ON VALIDATION SET
                    val_loss = torch.tensor(0., device=device)
                    val_f1 = torch.tensor(0., device=device)
                    for survey, species, tokens, y in tqdm(val_loader, desc="Validating"):
                        tokens = tokens.to(device).long()
                        species = species.to(device).float()
                        for i in range(len(y)):
                            y[i] = y[i].to(device)

                        logits = ema(y, tokens)
                        val_loss += compute_loss(logits, species, tokens).item()

                        samples = sample_fn(y).squeeze(1)

                        for i in range(samples.shape[0]):
                            pred = samples[i]
                            tp = (pred * species[i]).sum()
                            if tp > 0 :
                                val_f1 += 2 * tp / (species[i].sum() + pred.sum())

                    dist.all_reduce(val_f1, op=dist.ReduceOp.SUM)
                    dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)

                    val_f1 /= args.batch_size*len(val_loader)
                    val_loss /= dist.get_world_size()*len(val_loader)
                    val_f1 = val_f1.item()
                    val_loss = val_loss.item()
                    metrics["val/f1"] = val_f1
                    metrics["val/loss"] = val_loss



                    test_f1 = torch.tensor(0., device=device)
                    test_loss = torch.tensor(0., device=device)
                    for survey, species, tokens, y in tqdm(test_loader, desc="Testing"):
                        tokens = tokens.to(device).long()
                        species = species.to(device).float()
                        for i in range(len(y)):
                            y[i] = y[i].to(device)

                        logits = ema(y, tokens)
                        test_loss += compute_loss(logits, species, tokens).item()
                        samples = sample_fn(y).squeeze(1)
                        for i in range(samples.shape[0]):
                            pred = samples[i]
                            tp = (pred * species[i]).sum()
                            if tp > 0:
                                test_f1 += 2 * tp / (species[i].sum() + pred.sum())

                    dist.all_reduce(test_f1, op=dist.ReduceOp.SUM)
                    dist.all_reduce(test_loss, op=dist.ReduceOp.SUM)
                    test_f1 /= len(test_loader.dataset)
                    test_loss /= len(test_loader.dataset)
                    metrics["test/f1"] = test_f1.item()
                    metrics["test/loss"] = test_loss.item()

                    if rank == 0:
                        test_summary = f" | Test: Loss {metrics['test/loss']:.2e} - F1 {metrics['test/f1']:.3f}"
                        logger.info(f"Train : F1 {train_f1:.3f} | Val : Loss {val_loss:.2e} - F1: {val_f1:.3f}{test_summary}")
                        logger.info("Generating EMA samples done.")
                        if args.wandb:
                            utils.log(metrics, step=train_steps)
                    dist.barrier()                       

if __name__ == "__main__":

    config = load_config()
    training = config["Training"]
    data = config["Data"]
    data_args = data['Dataset Args']
    trained_instances = config["Trained Instances"]
    global_args = as_namespace({"data_args": data_args, **training, **data["Paths & Directories"], **data["Folds"], **data["Dataset"], **data_args})


    fold_list = [None]
    if global_args.test_path is None:
        if global_args.kfold_path is None:
            raise ValueError("Experiment should contain either a test or a kfold.")
        fold_list = list(set(p.read_csv(global_args.kfold_path)[global_args.index_fold]))

    data = prepare_data(global_args)

    init_main(global_args)
    
    if dist.get_rank() == 0:
        create_dir(global_args)

    try :
        for instance_name, instance in trained_instances.items():
            for i in range(len(fold_list)):
                args = deepcopy(global_args)
                args.model = instance["type"]
                args.model_args = instance.get("model_args", {})
                args.sampler = instance["sampler"]
                args.sampler_args = instance.get("sampler_args", {})
                args.ckpt = instance.get("ckpt")
                args.instance_name = instance_name
                args.fold = i
                args.fold_name = fold_list[i]
                main(args, data)
    finally:
        cleanup()