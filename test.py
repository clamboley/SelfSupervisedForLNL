from pathlib import Path
import os
import json
import argparse

import matplotlib.pyplot as plt


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained model on ImageNet"
    )

    # Data
    parser.add_argument("--data-dir", type=Path, help="path to dataset",
                        default="")

    # Noise configurations
    parser.add_argument('--noises', '-n', nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], type=float)
    parser.add_argument('--sparsities', '-s', nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], type=float)
    parser.add_argument('--listseed', nargs="+", default=[0, 1, 2, 3, 4], type=int)

    # Checkpoint
    parser.add_argument(
        "--exp-dir",
        default="./results",
        type=Path,
        metavar="DIR",
        help="path to checkpoint directory",
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
                        help='Temperature for NTXent loss')
    parser.add_argument("--batch-size", type=int, default=256,
                        help='Batch size used for SimCLR')

    # MLP
    parser.add_argument("--weights", default="freeze", type=str, choices=("finetune", "freeze"),
                        help="finetune or freeze resnet weights")
    parser.add_argument('--loss', default='crossentropy', type=str, choices=('crossentropy', 'elr'))

    # Plot
    parser.add_argument('-encoder-hist', default=False, action='store_true')

    parser.add_argument('-mlp-hist', default=False, action='store_true')
    parser.add_argument('-metric', default='acc1', type=str, choices=('loss', 'acc1', 'acc5'))

    parser.add_argument('-acc-table', default=False, action='store_true')

    return parser


def plot_encoder_training(args):
    results_path = {'vicreg': os.path.join("VICReg", f"{args.arch}_{args.arch_epochs}",
                                           f"V{args.std_coeff}_I{args.sim_coeff}_C{args.cov_coeff}"),
                    'simclr': os.path.join("SimCLR", f"{args.arch}_{args.arch_epochs}",
                                           f"T{args.temperature}_B{args.batch_size}")}
    path = Path(args.exp_dir / results_path[args.method] / "encoder")
    full_dict = {"epoch": [], "step": [], "loss": [], "time": [], "lr": []}

    with open(path / "stats.txt") as stats_file:
        for row in stats_file:
            try:
                tmp_dict = json.loads(row)
                for key in tmp_dict:
                    full_dict[key].append(tmp_dict[key])
            except ValueError:
                print("Skipping invalid line {}.".format(repr(row)))

    step_max = max(full_dict['step'])
    full_dict['step'] = [(step / step_max) * args.arch_epochs for step in full_dict['step']]

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title(f'{args.method.upper()} training loss with {args.arch} for {args.arch_epochs} epochs.')
    ax.set_xlabel('epoch')
    ax.set_ylabel('train loss')
    ax.plot(full_dict['step'], full_dict['loss'])


def plot_mlp_training(args):
    results_path = {'vicreg': os.path.join("VICReg", f"{args.arch}_{args.arch_epochs}",
                                           f"V{args.std_coeff}_I{args.sim_coeff}_C{args.cov_coeff}"),
                    'simclr': os.path.join("SimCLR", f"{args.arch}_{args.arch_epochs}",
                                           f"T{args.temperature}_B{args.batch_size}")}
    path = Path(args.exp_dir / results_path[args.method] / f"mlp_{args.weights}")

    train_dict = {'epoch': [], 'step': [], 'lr_backbone': [], 'lr_head': [], 'loss': [], 'time': []}
    test_dict = {'epoch': [], 'acc1': [], 'acc5': [], 'best_acc1': [], 'best_acc5': []}

    with open(
            path / f"seed{args.listseed[0]}_n{args.noises[0]}_s{args.sparsities[0]}_{args.loss}_stats.txt"
    ) as stats_file:
        for row in stats_file:
            try:
                tmp_dict = json.loads(row)
                if 'time' in tmp_dict:
                    for key in tmp_dict:
                        train_dict[key].append(tmp_dict[key])
                else:
                    for key in tmp_dict:
                        test_dict[key].append(tmp_dict[key])
            except ValueError:
                print("Skipping invalid line {}.".format(repr(row)))

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title(f'MLP training on top of {args.arch} trained for {args.arch_epochs} epochs. '
                 f'| seed: {args.listseed[0]} , N: {args.noises[0]} , S: {args.sparsities[0]}')
    ax.set_xlabel('epoch')

    if args.metric in ['acc1', 'acc5']:
        ax.set_ylabel(f'Test {args.metric}')
        ax.plot(test_dict['epoch'], test_dict[args.metric], label=args.metric)
        ax.plot(test_dict['epoch'], test_dict[f'best_{args.metric}'], ':', label=f'best_{args.metric}')
    else:
        ax.set_ylabel('Train loss')
        time_max = max(train_dict['time'])
        train_dict['time'] = [(t / time_max) * args.arch_epochs for t in train_dict['time']]
        ax.plot(train_dict['time'], train_dict[args.metric], label=args.metric)
    ax.legend(loc='right')


def show_acc_table(args):
    results_path = {'vicreg': os.path.join("VICReg", f"{args.arch}_{args.arch_epochs}",
                                           f"V{args.std_coeff}_I{args.sim_coeff}_C{args.cov_coeff}"),
                    'simclr': os.path.join("SimCLR", f"{args.arch}_{args.arch_epochs}",
                                           f"T{args.temperature}_B{args.batch_size}")}
    path = Path(args.exp_dir / results_path[args.method] / f"mlp_{args.weights}")

    print('-' * (10 + (7 * len(args.sparsities))))
    print('  NOISE |', end='')
    for i in range(len(args.sparsities) - 1):
        if i == 0:
            print('{}'.format(' SPARSITY'), end='')
        if i == 1:
            print(' ' * 5, end='')
        else:
            print(' ' * 7, end='')

    print('\n        |', end='')
    for spars in args.sparsities:
        print('{:^7.1f}'.format(spars), end='')
    print('\n' + '-' * (10 + (7 * len(args.sparsities))), end='')

    for noise in args.noises:
        print('\n {:^7.1f}|'.format(noise), end='')
        for spars in args.sparsities:
            mean_acc, count = 0.0, 0
            for seed in args.listseed:
                file_name = f'seed{seed}_n{noise}_s{spars}_{args.loss}_stats.txt'
                file_path = os.path.join(path, file_name)

                if os.path.isfile(file_path):
                    with open(file_path) as stats_file:
                        last_row = stats_file.readlines()[-1]
                        tmp_acc = json.loads(last_row)['best_acc1']
                    mean_acc += tmp_acc
                    count += 1

            if count != 0:
                # print('{:^7d}'.format(count), end='')
                print('{:^7.2f}'.format(mean_acc / count), end='')
            else:
                print('{:^7s}'.format('__.__'), end='')

    print('\n' + '-' * (10 + (7 * len(args.sparsities))), end='\n')


def test(args):

    if args.encoder_hist:
        plot_encoder_training(args)

    if args.mlp_hist:
        plot_mlp_training(args)

    if args.acc_table:
        show_acc_table(args)

    plt.show()


if __name__ == "__main__":
    parser = get_arguments()
    test(parser.parse_args())



