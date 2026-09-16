# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tkinter import font
import torch
from copy import deepcopy
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import os
import matplotlib.pyplot as plt

from models import MarginalModel, AutoRegModel, StochasticMixtureModel, TabularEmbedder
from torch.utils.data.distributed import DistributedSampler
import pandas as p
from dataset import SpeciesVocab, TabularDataset
from tqdm import tqdm
from config import as_namespace, load_config
from med_base import init_main, prepare_data, build_loaders, create_logger
from samplers import AutoRegSampler, MarginalSampler, CSVSampler


def cleanup():
    dist.destroy_process_group()




########################################################################################

def sample_model_scores(args, data):
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0

    vocab, species_ds, _, _, _ = data
    _, _, loader = build_loaders(args, data)
    

    sampler_kwargs = dict(args.sampler_args)
    sampler_kwargs["vocab"] = vocab

    if args.ckpt_folder is not None and args.ckpt_step != "None":
        if args.fold_name is None:
            ckpt_path = os.path.join(args.ckpt_folder, f"{args.ckpt_step:07d}.pt")
        else:
            ckpt_path = os.path.join(args.ckpt_folder, f"fold_{args.fold_name}/checkpoints/{args.ckpt_step:07d}.pt")

        assert os.path.isfile(ckpt_path), f'Could not find checkpoint at {ckpt_path}'
        load_state = torch.load(ckpt_path, map_location="cpu")
        args.model = load_state.get("model_type")
        args.model_args = load_state.get("model_args")

        model_kwargs = dict(args.model_args)
        model_kwargs["vocab"] = vocab
        model_kwargs.setdefault("env_emb_class", "TabularEmbedder")
        model_kwargs.setdefault("input_size", len(args.data_args["var_cols"]))
        if args.model == "AutoRegModel":
            model_kwargs.setdefault("max_length", species_ds.max_length)
        model = globals()[args.model](**model_kwargs).to(device)

        model.load_state_dict(load_state["ema"])

        model = model.to(device)
        model.eval()  # EMA model should always be in eval mode
        sampler_kwargs["model"] = model


    if args.sampler == "AutoRegSampler":
        sampler_kwargs["max_length"] = species_ds.max_length
    sample_fn = globals()[args.sampler](**sampler_kwargs).sample

    #################################################################
    




    if rank == 0:
        print(f"Sampling from {args.instance_name} Fold {args.fold}")


    with torch.no_grad():

        sample_f1 = torch.zeros(args.nb_sample, device=device)
        marginal_f1 = torch.zeros(args.nb_sample, device=device)

        for survey, species, tokens, y in tqdm(loader, desc="Sampling"):
            for i in range(len(y)):
                y[i] = y[i].to(device)
            species = species.to(device).float()
            samples = sample_fn(y, args.nb_sample, survey).squeeze(1)

            for i in range(samples.shape[0]):
                tgt = species[i]
                max_score = -1*torch.inf
                for j in range(samples.shape[1]):
                    pred = samples[i,j]
                    tp = (pred * tgt).sum()
                    score = 2 * tp / (tgt.sum() + pred.sum())

                    max_score = max(max_score, score)
                    sample_f1[j] += max_score


            marginal_prob = torch.zeros(species.shape, device=device)
            if args.sampler == "MarginalSampler":
                logits = model(y)

            for j in range(samples.shape[1]):
                for i in range(samples.shape[0]):
                    tgt = species[i]
                    max_score = -1*torch.inf
                    pred = samples[i,j]

                    if args.sampler == "MarginalSampler":
                        prob = torch.sigmoid(logits[i])
                    else :
                        prob = pred.float()

                    marginal_prob[i] += prob
                    norm_prob = marginal_prob[i] / (j+1)
                    marginal_rs = norm_prob.sum().round().long()
                    marginal_pred = torch.zeros_like(norm_prob)
                    id_species = torch.argsort(norm_prob, dim=-1, descending=True)
                    marginal_pred[id_species[:marginal_rs]] = 1
                    tp = (marginal_pred * tgt).sum()
                    if tp > 0 :
                        score = 2 * tp / (tgt.sum() + marginal_pred.sum())
                    marginal_f1[j] += score

                
    return sample_f1, marginal_f1

if __name__ == "__main__":

    config = load_config()
    data = config["Data"]
    data_args = data['Dataset Args']
    evaluated_instances = config["Evaluated Instances"]
    evaluation = config["Evaluation"]
    training = config["Training"]
    global_args = as_namespace({"data_args": data_args, "evaluated_instances": evaluated_instances,
    **evaluation, **data["Paths & Directories"], **data["Folds"], **data["Dataset"], **data_args, **training})

    global_args.num_workers = 0

    fold_list = [None]
    if global_args.test_path is None:
        if global_args.kfold_path is None:
            raise ValueError("Experiment should contain either a test or a kfold.")
        fold_list = list(set(p.read_csv(global_args.kfold_path)[global_args.index_fold]))

    data = prepare_data(global_args)
    
    init_main(global_args)
    rank = dist.get_rank()
    if rank == 0:
        plt.figure(figsize=(10, 6))
        plt.rcParams['font.family'] = 'Arial'
 

    ds_length = len(data[1] ) if global_args.test_path is None else len(data[2])
    try :
        for i, (instance_name, instance) in enumerate(evaluated_instances.items()):
            sample_scores = torch.zeros(global_args.nb_sample, device = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0)
            marginal_scores = torch.zeros(global_args.nb_sample, device = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0)
            for j in range(len(fold_list)):
                args = deepcopy(global_args)
                args.model_args = instance.get("model_args", {})
                args.sampler = instance["sampler"]
                args.sampler_args = instance.get("sampler_args", {})
                args.ckpt_folder = instance.get("ckpt_folder", None)
                args.ckpt_step = instance.get("ckpt_step", None)
                args.instance_name = instance_name
                args.fold = j
                args.fold_name = fold_list[j]
                scores = sample_model_scores(args, data)
                sample_scores += scores[0]
                marginal_scores += scores[1]
            dist.all_reduce(sample_scores, op=dist.ReduceOp.SUM)
            dist.all_reduce(marginal_scores, op=dist.ReduceOp.SUM)
            sample_scores /= ds_length
            marginal_scores /= ds_length

            if rank == 0:
                plt.plot(range(1, args.nb_sample + 1), sample_scores.cpu().numpy(), label= instance.get("label", instance_name), color = f"#{args.colors[i]}")
                if args.sampler == "MarginalSampler":
                    plt.plot(range(1, args.nb_sample + 1), marginal_scores.cpu().numpy(), color = f"#{args.colors[i]}", linestyle = "--", label = "Thresholded Baseline")
                print(f"Instance {instance_name} Fold {j} Sample F1: {sample_scores[-1].item():.4f} Marginal F1: {marginal_scores[-1].item():.4f}")

        if rank == 0:
            plt.xlabel("Number of samples n", fontsize=18, fontfamily='Arial')
            plt.ylabel("Best-of-n F1 Score", fontsize=18, fontfamily='Arial')
            xt = list(plt.xticks()[0])
            xt = sorted(set(xt + [1]))
            plt.xticks(xt, fontsize=14, fontfamily='Arial')
            plt.yticks(fontsize=14, fontfamily='Arial')
            plt.xlim(1, args.nb_sample)
            plt.grid()
            plt.legend(fontsize=18, loc="lower right", frameon=True, framealpha=0.9)
            plt.savefig("f1_curve.png")
            plt.show()

    finally:
        cleanup()