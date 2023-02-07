import sys
import os
sys.path.append(os.path.expanduser("~/repos/NeurOps/pytorch"))
import numpy as np
import torch
from collections import OrderedDict
import argparse

from neurops import *

from growprune import *

def transfer(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.task == "galaxy10":
        limit = 15 if args.reset_linear else 19
        model = ModVGG11(num_classes=10, avgpooldim=7).to(device)
        weights = torch.load("../../data/vgg11-8a719046.pth")
        renamed = OrderedDict()
        for i, (key, value) in enumerate(zip(model.state_dict(), weights.values())):
            if i <= limit:
                renamed[key] = value
            else:
                renamed[key] = model.state_dict()[key]
        model.load_state_dict(renamed)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()
        train_loader, validation_loader, test_loader = get_galaxy10_dataloaders(path_to_dir="../..", batch_size=args.batch_size, dim=224)

    test(model, train_loader, criterion, device=device)
    init_score = np.zeros(len(model.activations))
    for i in range(len(model)-1):
        tensor = model.activations[str(i)]
        print(torch.transpose(torch.transpose(tensor, 0, 1).reshape(tensor.shape[1], -1), 0, 1).shape)
        init_score[i] = effective_rank(model.activations[str(i)], limit_ratio = 10)
        print(f"Layer {i} has effective rank {init_score[i]}")

    for epoch in range(args.epochs):
        test(model, validation_loader, criterion, device=device)
        train(model, train_loader, optimizer, criterion, epochs=1, val_loader=validation_loader, device=device)
        for i in range(len(model)-1):
            max_rank = model[i].out_features if i > model.conversion_layer else model[i].out_channels
            score = effective_rank(model.activations[str(i)], limit_ratio = 100)
            num_to_prune = max(int(0.9*init_score[i])-score, 0)
            scores = svd_score(model.activations[str(i)], limit_ratio = 100)
            to_prune = np.argsort(scores.cpu().detach().numpy())[:num_to_prune]
            to_add = max(score-int(0.97*init_score[i]), 0)
            print("Layer {} score: {}/{}, neurons to prune: {}, # neurons to add: {}".format(i, score, max_rank, to_prune, to_add))
            model.prune(i, to_prune, optimizer=optimizer)
            model.grow(i, to_add, fanin_weights="iterative_orthogonalization", 
                                optimizer=optimizer)





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run diffentiable equivariance-aware NAS')
    parser.add_argument('--task', "-t", type=str, default="galaxy10", help='task')
    parser.add_argument('--reset_linear', action='store_true', default=False, help='start linear layers w/ random init')
    parser.add_argument('--pruning_threshold', type=float, default=0.9, help='pruning threshold')
    parser.add_argument('--growing_threshold', type=float, default=0.97, help='growing threshold')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    transfer(args)