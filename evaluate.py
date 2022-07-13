# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib

from torch import nn, optim, Tensor
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch

import resnet


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset", default="")

    # Label noise
    parser.add_argument("--noise", default=0.0, type=float, help="label noise")
    parser.add_argument("--sparsity", default=0.0, type=float, help="label sparsity")
    parser.add_argument("--seed", default=0, type=int, help="seed used for noise generation")

    # Checkpoint
    parser.add_argument(
        "--exp-dir",
        default="./results",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
    )
    parser.add_argument(
        "--print-freq", default=50, type=int, metavar="N", help="print frequency"
    )

    # Backbone
    parser.add_argument("--method", type=str, default="vicreg", choices=('vicreg', 'simclr'),
                        help='Self-Supervised Learning algorithm used to create encoder')
    parser.add_argument("--arch", type=str, default="resnet18",
                        choices=('resnet18', 'resnet34', 'resnet50', 'resnet101',
                                 'resnet50x2', 'resnet50x4', 'resnet50x5', 'resnet200x2'),
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--arch-epochs", type=int, default=2000)

    # SSL hyperparameters
    parser.add_argument("--std-coeff", "-V", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--sim-coeff", "-I", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--cov-coeff", "-C", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
    parser.add_argument("--temperature", "-T", type=float, default=0.5,
                        help='Covariance regularization loss coefficient')
    parser.add_argument("--arch-batch-size", default=256, type=int, metavar="N", help="SimCLR batch size")

    # Optim
    parser.add_argument('--loss', default='crossentropy', type=str, choices=('crossentropy', 'elr'))
    parser.add_argument('--stopping', default=20, type=int, help="epochs before early stopping")
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--batch-size", default=256, type=int, metavar="N", help="mini-batch size"
    )
    parser.add_argument(
        "--lr-backbone",
        default=0.0,
        type=float,
        metavar="LR",
        help="backbone base learning rate",
    )
    parser.add_argument(
        "--lr-head",
        default=0.1,
        type=float,
        metavar="LR",
        help="classifier base learning rate",
    )
    parser.add_argument(
        "--weight-decay", default=1e-6, type=float, metavar="W", help="weight decay"
    )
    parser.add_argument(
        "--weights",
        default="freeze",
        type=str,
        choices=("finetune", "freeze"),
        help="finetune or freeze resnet weights",
    )

    # Running
    parser.add_argument(
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loader workers",
    )

    return parser


def main():
    parser = get_arguments()
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    if "SLURM_JOB_ID" in os.environ:
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
    # single-node distributed training
    args.rank = 0
    args.dist_url = f"tcp://localhost:{random.randrange(49152, 65535)}"
    args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    torch.distributed.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    results_path = {'vicreg': os.path.join("VICReg", f"{args.arch}_{args.arch_epochs}",
                                           f"V{args.std_coeff}_I{args.sim_coeff}_C{args.cov_coeff}"),
                    'simclr': os.path.join("SimCLR", f"{args.arch}_{args.arch_epochs}",
                                           f"T{args.temperature}_B{args.arch_batch_size}")}
    encoder_dir = Path(args.exp_dir / results_path[args.method] / "encoder")
    mlp_dir = Path(args.exp_dir / results_path[args.method] / f"mlp_{args.weights}")

    if args.rank == 0:
        mlp_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(mlp_dir / f'seed{args.seed}_n{args.noise}_s{args.sparsity}_{args.loss}_stats.txt',
                          "w", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    state_dict = torch.load(encoder_dir / "backbone.pth", map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
        state_dict = {
            key.replace("module.backbone.", ""): value
            for (key, value) in state_dict.items()
        }
    backbone.load_state_dict(state_dict, strict=False)

    head = nn.Sequential(
            nn.Linear(embedding, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 10)
        )
    model = nn.Sequential(backbone, head)
    model.cuda(gpu)

    if args.weights == "freeze":
        backbone.requires_grad_(False)
        head.requires_grad_(True)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    param_groups = [dict(params=head.parameters(), lr=args.lr_head)]
    if args.weights == "finetune":
        param_groups.append(dict(params=backbone.parameters(), lr=args.lr_backbone))
    optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_head * 0.1)

    start_epoch = 0
    best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code
    traindir = os.path.join(args.data_dir, 'cifar10_dataset', 'base_dataset', 'train')
    testdir = os.path.join(args.data_dir, 'cifar10_dataset', 'base_dataset', 'test')
    train_transfo = transforms.Compose(
        [
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]
    )
    test_transfo = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]
    )

    train_dataset = ImageFolderModified(traindir, train_transfo)
    test_dataset = ImageFolderModified(testdir, test_transfo)

    labels_path = os.path.join(args.data_dir, 'cifar10_dataset', 'noisy_labels_{}'.format(args.seed),
                               'noisylabels_noise{}_sparsity{}.json'.format(args.noise, args.sparsity))
    with open(labels_path) as rf:
        tmp = json.load(rf)
    train_labels_dict = {}
    for k in tmp:
        train_labels_dict[os.path.join(args.data_dir, k)] = tmp[k]
    train_dataset.imgs = [(fn, train_labels_dict[fn]) for fn, _ in train_dataset.imgs]
    train_dataset.samples = train_dataset.imgs

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(
        batch_size=args.batch_size // args.world_size,
        num_workers=args.workers,
        pin_memory=True,
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

    str_loss = args.loss
    if str_loss == 'crossentropy':
        criterion = nn.CrossEntropyLoss().cuda(gpu)
    else:
        criterion = ELRlossRunningAvg(device=gpu, num_examp=len(train_dataset), lambda_elr=3, beta=0.7)

    count = 0
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # train
        if args.weights == "finetune":
            model.train()
        elif args.weights == "freeze":
            model.eval()
        else:
            assert False

        train_sampler.set_epoch(epoch)

        for step, (images, target, idx) in enumerate(
            train_loader, start=epoch * len(train_loader)
        ):
            output = model(images.cuda(gpu, non_blocking=True))
            if str_loss == 'crossentropy':
                loss = criterion(output, target.cuda(gpu, non_blocking=True))
            else:
                loss = criterion(output, target.cuda(gpu, non_blocking=True), idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                torch.distributed.reduce(loss.div_(args.world_size), 0)
                if args.rank == 0:
                    pg = optimizer.param_groups
                    lr_head = pg[0]["lr"]
                    lr_backbone = pg[1]["lr"] if len(pg) == 2 else 0
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        lr_backbone=lr_backbone,
                        lr_head=lr_head,
                        loss=loss.item(),
                        time=int(time.time() - start_time),
                    )
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)

        scheduler.step()

        # evaluate
        model.eval()
        if args.rank == 0:
            top1 = AverageMeter("Acc@1")
            top5 = AverageMeter("Acc@5")
            with torch.no_grad():
                for images, target, _ in test_loader:
                    output = model(images.cuda(gpu, non_blocking=True))
                    acc1, acc5 = accuracy(
                        output, target.cuda(gpu, non_blocking=True), topk=(1, 5)
                    )
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            stats = dict(
                epoch=epoch,
                acc1=top1.avg,
                acc5=top5.avg,
                best_acc1=best_acc.top1,
                best_acc5=best_acc.top5,
            )
            print(json.dumps(stats))
            print(json.dumps(stats), file=stats_file)

            if top1.avg == best_acc.top1:
                count = 0
            else:
                count += 1
                if count > args.stopping:
                    break


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ELRlossRunningAvg(nn.Module):
    def __init__(self, device, num_examp, num_classes=10, lambda_elr=3, beta=0.7):
        super(ELRlossRunningAvg, self).__init__()
        self.num_classes = num_classes
        self.target = torch.zeros(num_examp, self.num_classes).to(device)
        self.beta = beta
        self.lambda_elr = lambda_elr

    def forward(self, output, label, index):
        y_pred = F.softmax(output, dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1 - self.beta) * (y_pred_/y_pred_.sum(dim=1,
                                                                                                     keepdims=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + self.lambda_elr * elr_reg

        return final_loss


# Image folder returning "index" for the ELR loss
class ImageFolderModified(datasets.ImageFolder):
    def __init__(self, root, transform):
        super(ImageFolderModified, self).__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index


if __name__ == "__main__":
    main()
