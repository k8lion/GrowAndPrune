import csv
import itertools
import os
import time
from collections import defaultdict
import copy

from pathlib import Path
from datetime import date
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.optim as optim
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
import wandb
import pandas as pd
from PIL import Image

import utils
from dataset import corruption_types, get_loaders


@torch.no_grad()
def test(model, loader, criterion, cfg):
    model.eval()
    all_test_corrects = []
    total_loss = 0.0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        if cfg.args.flip_labels:
            y = 9-y  
        logits = model(x)
        loss = criterion(logits, y)
        all_test_corrects.append(torch.argmax(logits, dim=-1) == y)
        total_loss += loss
    acc = torch.cat(all_test_corrects).float().mean().detach().item()
    total_loss = total_loss / len(loader)
    total_loss = total_loss.detach().item()
    return acc, total_loss


def compute_layerwise_metrics(model, loader):
    # Only considering "weight" gradients for simplicity
    layer_names = [
        n for n, _ in model.named_parameters() if "bn" not in n
    ]  # and "bias" not in n]

    metrics = defaultdict(list)
    partial_loader = itertools.islice(loader, 5)
    xent_grads, entropy_grads = [], []
    for x, y in partial_loader:
        x, y = x.cuda(), y.cuda()
        logits = model(x)

        loss_xent = F.cross_entropy(logits, y)
        grad_xent = torch.autograd.grad(
            outputs=loss_xent, inputs=model.parameters(), retain_graph=True
        )
        xent_grads.append([g.detach() for g in grad_xent])

        # Entropy of predictions. Can calculate without labels.
        loss_entropy = Categorical(logits=logits).entropy().mean()
        grad_entropy = torch.autograd.grad(
            outputs=loss_entropy, inputs=model.parameters(), retain_graph=True
        )
        entropy_grads.append([g.detach() for g in grad_entropy])

    def get_grad_norms(model, grads):
        _metrics = defaultdict(list)
        grad_norms, rel_grad_norms = [], []
        for (name, param), grad in zip(model.named_parameters(), grads):
            if name not in layer_names:
                continue
            _metrics["grad_norm"].append(torch.norm(grad).item())
            _metrics["rel_grad_norm"].append(
                torch.norm(grad).item() / torch.norm(param).item()
            )
            _metrics["grad_abs"].append(grad.abs().mean().item())
            _metrics["rel_grad_abs"].append((grad.abs() / (param.abs() + 1e-6)).mean().item())
        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(model, xent_grad)
        for k, v in xent_grad_metrics.items():
            metrics[f"xent_{k}"].append(v)
    for entropy_grad in entropy_grads:
        ent_grad_metrics = get_grad_norms(model, entropy_grad)
        for k, v in ent_grad_metrics.items():
            metrics[f"ent_{k}"].append(v)

    num_pointwise = min(10, loader.batch_size)
    pt_xent_grads, pt_ent_grads = [], []
    x, y = next(iter(loader))
    x, y = x.cuda(), y.cuda()
    # y = 9-y
    logits = model(x)
    loss_xent_pointwise = F.cross_entropy(logits, y, reduction="none")[:num_pointwise]
    for _loss in loss_xent_pointwise:
        grad_xent_pt = torch.autograd.grad(
            outputs=_loss, inputs=model.parameters(), retain_graph=True
        )
        pt_xent_grads.append([g.detach() for g in grad_xent_pt])
    loss_ent_pointwise = Categorical(logits=logits).entropy()[:num_pointwise]
    for _loss in loss_ent_pointwise:
        grad_ent_pt = torch.autograd.grad(
            outputs=_loss, inputs=model.parameters(), retain_graph=True
        )
        pt_ent_grads.append([g.detach() for g in grad_ent_pt])

    def get_pointwise_grad_norms(model, grads):
        all_cosine_sims = []
        for grads1, grads2 in itertools.combinations(grads, 2):
            cosine_sims = []
            for (name, _), g1, g2 in zip(model.named_parameters(), grads1, grads2):
                if name not in layer_names:
                    continue
                cosine_sims.append(
                    F.cosine_similarity(g1.flatten(), g2.flatten(), dim=0).item()
                )
            all_cosine_sims.append(cosine_sims)
        return all_cosine_sims

    metrics["xent_pairwise_cosine_sim"] = get_pointwise_grad_norms(model, pt_xent_grads)
    metrics["ent_pairwise_cosine_sim"] = get_pointwise_grad_norms(model, pt_ent_grads)

    for k, v in metrics.items():
        average_layerwise_metric = np.array(v).mean(0)
        plt.plot(range(len(average_layerwise_metric)), average_layerwise_metric, label=k)
        plt.xlabel("Layer")
        plt.title(f"{k}")
        wandb.log({f"plots/{k}": wandb.Image(plt)}, commit=False)
        plt.cla()


def get_lr_weights(model, loader, cfg):
    # Only considering "weight" gradients for simplicity
    layer_names = [
        n for n, _ in model.named_parameters() if "bn" not in n
    ]  # and "bias" not in n]

    metrics = defaultdict(list)
    average_metrics = defaultdict(float)
    partial_loader = itertools.islice(loader, 5)
    xent_grads, entropy_grads = [], []
    for x, y in partial_loader:
        x, y = x.cuda(), y.cuda()
        if cfg.args.flip_labels:
            y = 9-y  # Reverse labels; quick way to simulate last-layer setting
        logits = model(x)

        loss_xent = F.cross_entropy(logits, y)
        grad_xent = torch.autograd.grad(
            outputs=loss_xent, inputs=model.parameters(), retain_graph=True
        )
        xent_grads.append([g.detach() for g in grad_xent])

    def get_grad_norms(model, grads, cfg):
        _metrics = defaultdict(list)
        grad_norms, rel_grad_norms = [], []
        for (name, param), grad in zip(model.named_parameters(), grads):
            if name not in layer_names:
                continue
            if cfg.args.auto_tune == 'eb-criterion':
                tmp = (grad*grad) / (torch.var(grad, dim=0, keepdim=True)+1e-8)
                _metrics[name] = tmp.mean().item()
            else:
                _metrics[name] = torch.norm(grad).item() / torch.norm(param).item()

        return _metrics

    for xent_grad in xent_grads:
        xent_grad_metrics = get_grad_norms(model, xent_grad, cfg)
        for k, v in xent_grad_metrics.items():
            metrics[k].append(v)
    for k, v in metrics.items():
        average_metrics[k] = np.array(v).mean(0)
    return average_metrics

@torch.no_grad()
def plot_loss_surface(orig_model, model, loader, criterion, cfg, tune_option):
    model.eval()
    orig_model.eval()
    alphas = np.arange(0, 1.1, 0.1)
    losses = []
    orig_params = list(orig_model.parameters())
    tuned_params = list(model.parameters())
    for alpha in alphas:
        total_loss = 0.0
        alpha_model = copy.deepcopy(orig_model)
        for i, p in enumerate(alpha_model.parameters()):
            p.data.copy_(alpha * orig_params[i] + (1-alpha) * tuned_params[i].cpu())
        alpha_model = alpha_model.cuda()  
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            if cfg.args.flip_labels:
                y = 9-y  # Reverse labels; quick way to simulate last-layer setting
            logits = alpha_model(x)
            loss = criterion(logits, y)
            total_loss += loss
        total_loss = total_loss / len(loader)
        total_loss = total_loss.detach().item()
        losses.append(total_loss)
    plt.plot(alphas, losses)
    plt.xlabel("Alpha")
    plt.ylabel("Train loss")
    plt.title(f"Loss surface tune{tune_option}_auto{cfg.args.auto_tune}")
    plt.savefig(f"{cfg.args.log_dir}/tune{tune_option}_auto{cfg.args.auto_tune}_valloss_surface.png")
    plt.close()


def train(model, loader, criterion, opt, cfg, orig_model=None):
    all_train_corrects = []
    total_loss = 0.0
    magnitudes = defaultdict(float)

    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        if cfg.args.flip_labels:
            y = 9-y 
        logits = model(x)
        loss = criterion(logits, y)
        all_train_corrects.append(torch.argmax(logits, dim=-1) == y)
        total_loss += loss
        if cfg.args.l1_reg:
            l1_reg = 0
            orig_params = list(orig_model.parameters())
            i = 0
            for param in model.parameters():
                l1_reg += torch.norm(orig_params[i] - param.cpu(), p=1)
                i += 1
            loss += cfg.args.l1_reg * l1_reg

        opt.zero_grad()
        loss.backward()
        opt.step()

    acc = torch.cat(all_train_corrects).float().mean().detach().item()
    total_loss = total_loss / len(loader)
    total_loss = total_loss.detach().item()
    return acc, total_loss, magnitudes


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    cfg.args.log_dir = Path.cwd()
    cfg.args.log_dir = os.path.join(
        cfg.args.log_dir, "results", cfg.data.dataset_name, date.today().strftime("%Y.%m.%d"), cfg.args.auto_tune
    )
    print(f"Log dir: {cfg.args.log_dir}")
    os.makedirs(cfg.args.log_dir, exist_ok=True)

    tune_options = [
        "first_two_block",
        "second_block",
        "third_block",
        "last",
        "all",
    ]
    if cfg.data.dataset_name == "imagenet-3dcc" or cfg.data.dataset_name == "imagenet-c":
        tune_options.append("fourth_block")
    if cfg.args.layerwise_metrics:
        tune_options = ["all"]
    if cfg.args.auto_tune != 'none' or cfg.args.l1_reg > 0:
        tune_options = ["all"]
        cfg.args.epochs_finetune = 0
#     if cfg.data.dataset_name == "imagenet-3dcc": tune_options = ["first_second"]
    if cfg.args.epochs == 0: tune_options = ['all']
    corruption_types = cfg.data.corruption_types
    for corruption_type in corruption_types:
        cfg.wandb.exp_name = f"{cfg.data.dataset_name}_corruption{corruption_type}"
        if cfg.wandb.use:
            utils.setup_wandb(cfg)
        utils.set_seed_everywhere(cfg.args.seed)
        loaders = get_loaders(cfg, corruption_type, cfg.data.severity)

        for tune_option in tune_options:
            tune_metrics = defaultdict(list)
            if cfg.data.dataset_name == "imagenet-3dcc" or cfg.data.dataset_name == "imagenet-c":
                    lr_wd_grid = [
                    (1e-4, 1e-4),
                    (1e-5, 1e-4),
                    ]
            else:
                if tune_option == "last":
                    lr_wd_grid = [
                        (1e-1, 1e-4),
                        (1e-2, 1e-4),
                        (1e-3, 1e-4),
                        (1e-4, 1e-4),
                    ]
                else:
                    lr_wd_grid = [
                        (1e-2, 1e-4),
                        (1e-3, 1e-4),
                        (1e-4, 1e-4),
                        (1e-5, 1e-4),
                    ]
            for lr, wd in lr_wd_grid:
                dataset_name = (
                    "imagenet"
                    if cfg.data.dataset_name == "imagenet-3dcc" or cfg.data.dataset_name == "imagenet-c"
                    else cfg.data.dataset_name
                )
                model = load_model(
                    cfg.data.model_name,
                    cfg.user.ckpt_dir,
                    dataset_name,
                    ThreatModel.corruptions,
                )

                orig_model = copy.deepcopy(model)
                model = model.cuda()

                # For cifar10
                if cfg.data.dataset_name == "cifar10" or cfg.data.dataset_name == "cifar100":
                    tune_params_dict = {
                        "all": [model.parameters()],
                        "first_two_block": [
                            model.conv1.parameters(),
                            model.block1.parameters(),
                        ],
                        "second_block": [
                            model.block2.parameters(),
                        ],
                        "third_block": [
                            model.block3.parameters(),
                        ],
                        "last": [model.fc.parameters()],
                    }
                elif cfg.data.dataset_name == "imagenet-3dcc" or cfg.data.dataset_name == "imagenet-c":
                    tune_params_dict = {
                        "all": [model.model.parameters()],
                        "first_second": [
                            model.model.conv1.parameters(),
                            model.model.layer1.parameters(),
                            model.model.layer2.parameters(),
                        ],
                        "first_two_block": [
                            model.model.conv1.parameters(),
                            model.model.layer1.parameters(),
                        ],
                        "second_block": [
                            model.model.layer2.parameters(),
                        ],
                        "third_block": [
                            model.model.layer3.parameters(),
                        ],
                        "fourth_block": [
                            model.model.layer4.parameters(),
                        ],
                        "last": [model.model.fc.parameters()],
                    }

                params_list = list(itertools.chain(*tune_params_dict[tune_option]))
                
                # if doing grad noise experiment
                if cfg.args.grad_noise != 'none':
                    # Add noise to grad noise layer
                    print(f"Adding noise to {cfg.args.grad_noise} layer")
                    with torch.no_grad():
                        if tune_option == cfg.args.grad_noise:
                            params_noise = [p for p in params_list if p.requires_grad]
                        else:
                            params_noise = list(itertools.chain(*tune_params_dict[cfg.args.grad_noise]))
                            #tune_params_dict[cfg.args.grad_noise]
                        print(f"Number of noisy params: {sum(p.numel() for p in params_noise)}")
                        #weight = 0.003
                        if cfg.args.grad_noise == 'first_two_block':
                            weight = 0.01
                        elif cfg.args.grad_noise == 'last':
                            weight = 8.0
                        elif cfg.args.grad_noise == 'second_block':
                            weight = 0.005
                        else:
                            weight = 0.003
                        for p in params_noise:
                            p += torch.rand(size=p.size()).cuda() * weight
                
                opt = optim.Adam(params_list, lr=lr, weight_decay=wd)
                N = sum(p.numel() for p in params_list if p.requires_grad)

                print(
                    f"\nTrain mode={cfg.args.train_mode}, using {cfg.args.train_n} corrupted images for training"
                )
                print(
                    f"Re-training {tune_option} ({N} params). lr={lr}, wd={wd}. Corruption {corruption_type}"
                )
                

                criterion = F.cross_entropy
                layer_weights = [0 for layer, _ in model.named_parameters() if 'bn' not in layer]
                layer_names = [layer for layer, _ in model.named_parameters() if 'bn' not in layer]
                for epoch in range(1, cfg.args.epochs + 1):
                    if cfg.args.train_mode == "train":
                        model.train()
                    if (
                        cfg.args.layerwise_metrics and epoch == 1
                    ):  # just compute near start of training
                        assert cfg.wandb.use
                        compute_layerwise_metrics(model, loaders["train"])
                    if cfg.args.auto_tune != 'none': # and (epoch-1)%3 == 0: 
                        if cfg.args.auto_tune == 'layer':
                            weights = get_lr_weights(model, loaders["train"], cfg) #get average grad norm/weight norm for each tensor over 5 batches
                            max_weight = max(weights.values())
                            for k, v in weights.items(): 
                                weights[k] = v / max_weight
                            layer_weights = [sum(x) for x in zip(layer_weights, weights.values())]
                            tune_metrics['layer_weights'] = layer_weights
                            params = defaultdict()
                            for n, p in model.named_parameters():
                                if "bn" not in n:
                                    params[n] = p 
                            params_weights = []
                            for param, weight in weights.items():
                                params_weights.append({"params": params[param], "lr": weight*lr})
                            opt = optim.Adam(params_weights, lr=lr, weight_decay=wd)
                        elif cfg.args.auto_tune == 'block':
                            # Choose which layers to tune and how much, go by blocks
                            if cfg.data.dataset_name == "cifar10" or cfg.data.dataset_name == "cifar100":
                                blocks ={"block1": list(model.conv1.parameters()) + list(model.block1.parameters()),
                                            "block2": list(model.block2.parameters()),
                                            "block3": list(model.block3.parameters()),
                                            "fc": list(model.fc.parameters()),
                                        }
                        elif cfg.args.auto_tune == 'eb-criterion':
                            # Go by individual layers
                            weights = get_lr_weights(model, loaders["train"], cfg)
                            print(f"Epoch {epoch}, autotuning weights {min(weights.values()), max(weights.values())}")
                            tune_metrics['max_weight'].append(max(weights.values()))
                            tune_metrics['min_weight'].append(min(weights.values()))
                            print(weights.values())
                            for k, v in weights.items(): 
                                weights[k] = 0.0 if v < 0.95 else 1.0 #1.02
                            print("weight values", weights.values())
                            layer_weights = [sum(x) for x in zip(layer_weights, weights.values())]
                            tune_metrics['layer_weights'] = layer_weights
                            params = defaultdict()
                            for n, p in model.named_parameters():
                                if "bn" not in n:
                                    params[n] = p 
                            params_weights = []
                            for k, v in params.items():
                                if k in weights.keys():
                                    params_weights.append({"params": params[k], "lr": weights[k]*lr})
                                else:
                                    params_weights.append({"params": params[k], "lr": 0.0})
                            opt = optim.Adam(params_weights, lr=lr, weight_decay=wd)
                            
                        elif cfg.args.auto_tune == 'gradual_first':
                            # Gradually unfreeze layers
                            if epoch == 0:
                                params_list = list(itertools.chain(*tune_params_dict["first_two_block"]))
                            if epoch == cfg.args.epochs // 5:
                                use_params = tune_params_dict["first_two_block"] + tune_params_dict["second_block"]
                                params_list = list(itertools.chain(*use_params))
                            if epoch == 2*cfg.args.epochs // 5:
                                use_params = tune_params_dict["first_two_block"] + tune_params_dict["second_block"] + tune_params_dict["third_block"] 
                                params_list = list(itertools.chain(*use_params))
                            if epoch == 3*cfg.args.epochs // 5:
                                use_params = tune_params_dict["first_two_block"] + tune_params_dict["second_block"] + tune_params_dict["third_block"] + tune_params_dict["fourth_block"] 
                                params_list = list(itertools.chain(*use_params))
                            if epoch == 4*cfg.args.epochs // 5:
                                params_list = list(itertools.chain(model.model.parameters()))
                            opt = optim.Adam(params_list, lr=lr, weight_decay=wd)
                            
                        elif cfg.args.auto_tune == 'gradual_last':
                            # Gradually unfreeze layers
                            if epoch == 0:
                                params_list = list(itertools.chain(*tune_params_dict["last"]))
                            if epoch == cfg.args.epochs // 5:
                                use_params = tune_params_dict["last"] + tune_params_dict["fourth_block"]
                                params_list = list(itertools.chain(*use_params))
                            if epoch == 2*cfg.args.epochs // 5:
                                use_params = tune_params_dict["last"] + tune_params_dict["fourth_block"] + tune_params_dict["third_block"] 
                                params_list = list(itertools.chain(*use_params))
                            if epoch == 3*cfg.args.epochs // 5:
                                use_params = tune_params_dict["last"] + tune_params_dict["second_block"] + tune_params_dict["third_block"] + tune_params_dict["fourth_block"] 
                                params_list = list(itertools.chain(*use_params))
                            if epoch == 4*cfg.args.epochs // 5:
                                params_list = list(itertools.chain(model.model.parameters()))
                            opt = optim.Adam(params_list, lr=lr, weight_decay=wd)
                            
                        else:
                            # Log rough fraction of parameters being tuned
                            no_weight = 0
                            for elt in params_weights:
                                if elt['lr'] == 0.:
                                    no_weight += elt['params'][0].flatten().shape[0]
                            total_params = sum(p.numel() for p in model.parameters())
                            tune_metrics['frac_params'].append((total_params-no_weight)/total_params)
                            print(f"Tuning {(total_params-no_weight)} out of {total_params} total")
                        
                    acc_tr, loss_tr, grad_magnitudes = train(
                        model, loaders["train"], criterion, opt, cfg, orig_model=orig_model
                    )
                    acc_te, loss_te = test(model, loaders["test"], criterion, cfg)
                    acc_val, loss_val = test(model, loaders["val"], criterion, cfg)
                    tune_metrics["acc_train"].append(acc_tr)
                    tune_metrics["acc_val"].append(acc_val)
                    tune_metrics["acc_te"].append(acc_te)
                    log_dict = {
                        f"{tune_option}/train/acc": acc_tr,
                        f"{tune_option}/train/loss": loss_tr,
                        f"{tune_option}/val/acc": acc_val,
                        f"{tune_option}/val/loss": loss_val,
                        f"{tune_option}/test/acc": acc_te,
                        f"{tune_option}/test/loss": loss_te,
                    }
                    print(f"Epoch {epoch:2d} Train acc: {acc_tr:.4f}, Val acc: {acc_val:.4f}")
                    
                    # More logging
                    l1_reg = 0
                    l2_reg = 0
                    orig_params = list(orig_model.parameters())
                    i = 0
                    for param in model.parameters():
                        l1_reg += torch.norm(orig_params[i] - param.cpu(), p=1)
                        l2_reg += torch.norm(orig_params[i] - param.cpu(), p=2)
                        i += 1
                    tune_metrics['l1_reg'].append((l1_reg*0.001).detach().numpy())
                    tune_metrics['l2_reg'].append((l2_reg*0.001).detach().numpy())

                    if cfg.wandb.use:
                        wandb.log(log_dict)

                ft_opt = optim.SGD(
                    model.parameters(), lr=cfg.args.lr_finetune, weight_decay=wd
                )
                print(f"Fine-tuning entire network. lr={cfg.args.lr_finetune}")
                for epoch in range(
                    cfg.args.epochs + 1, cfg.args.epochs_finetune + cfg.args.epochs + 1
                ):
                    if cfg.args.train_mode == "train":
                        model.train()
                    if (
                        cfg.args.layerwise_metrics and epoch == cfg.args.epochs + 1
                    ):  # just compute near start of training
                        assert cfg.wandb.use
                        compute_layerwise_metrics(model, loaders["train"])
                    acc_tr, loss_tr, grad_magnitudes = train(
                        model, loaders["train"], criterion, ft_opt, cfg
                    )
                    acc_te, loss_te = test(model, loaders["test"], criterion, cfg)
                    acc_val, loss_val = test(model, loaders["val"], criterion, cfg)
                    tune_metrics["acc_train"].append(acc_tr)
                    tune_metrics["acc_val"].append(acc_val)
                    tune_metrics["acc_te"].append(acc_te)
                    log_dict = {
                        f"{tune_option}/train/acc": acc_tr,
                        f"{tune_option}/train/loss": loss_tr,
                        f"{tune_option}/val/acc": acc_val,
                        f"{tune_option}/val/loss": loss_val,
                        f"{tune_option}/test/acc": acc_te,
                        f"{tune_option}/test/loss": loss_te,
                    }
                    print(f"Epoch {epoch:2d} Train acc: {acc_tr:.4f}, Val acc: {acc_val:.4f}")
                    if cfg.wandb.use:
                        wandb.log(log_dict)

                tune_metrics["lr_tested"].append(lr)
                tune_metrics["wd_tested"].append(wd)
            
            # Get test acc according to best val acc
            best_run_idx = np.argmax(np.array(tune_metrics["acc_val"]))
            best_testacc = tune_metrics["acc_te"][best_run_idx]
            best_lr_wd = best_run_idx // (cfg.args.epochs_finetune + cfg.args.epochs)

            print(
                f"Best epoch: {best_run_idx % (cfg.args.epochs_finetune + cfg.args.epochs)}, Test Acc: {best_testacc}"
            )

            data = {
                "corruption_type": corruption_type,
                "train_mode": cfg.args.train_mode,
                "tune_option": tune_option,
                "auto_tune": cfg.args.auto_tune,
                "grad_noise": cfg.args.grad_noise,
                "train_n": cfg.args.train_n,
                "seed": cfg.args.seed,
                "l1_reg": cfg.args.l1_reg,
                "lr": tune_metrics["lr_tested"][best_lr_wd],
                "wd": tune_metrics["wd_tested"][best_lr_wd],
                "val_acc": tune_metrics["acc_val"][best_run_idx],
                "best_testacc": best_testacc,
                "flip_labels": cfg.args.flip_labels,
            }

            recorded = False
            fieldnames = data.keys()
            csv_file_name = f"{cfg.args.log_dir}/results_seed{cfg.args.seed}.csv"
            write_header = True if not os.path.exists(csv_file_name) else False
            while not recorded:
                try:
                    with open(csv_file_name, "a") as f:
                        csv_writer = csv.DictWriter(f, fieldnames=fieldnames, restval=0.0)
                        if write_header:
                            csv_writer.writeheader()
                        csv_writer.writerow(data)
                    recorded = True
                except:
                    time.sleep(5)


if __name__ == "__main__":
    main()
