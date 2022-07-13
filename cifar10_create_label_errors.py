import os
import numpy as np
import json
import pickle

import torchvision
from torchvision import transforms

from cleanlab import noise_generation

import seaborn as sns
import matplotlib.pyplot as plt


data_path_base = os.path.join('cifar10_dataset', 'base_dataset')

# Create json with train labels
train_dataset = torchvision.datasets.ImageFolder(os.path.join(data_path_base, 'train'))
d = dict(train_dataset.imgs)
with open(os.path.join(data_path_base, "train_filename2label.json"), 'w') as wf:
    wf.write(json.dumps(d, indent=4))

# Create json with test labels
test_dataset = torchvision.datasets.ImageFolder(os.path.join(data_path_base, 'test'))
d = dict(test_dataset.imgs)
with open(os.path.join(data_path_base, "test_filename2label.json"), 'w') as wf:
    wf.write(json.dumps(d, indent=4))

# Create noisy labels for cifar10
train_dataset = torchvision.datasets.ImageFolder(
    root=os.path.join(data_path_base, 'train'),
    transform=transforms.ToTensor()
)

y_true = train_dataset.targets
K = 10

for seed in range(5):
    data_path_noisy = os.path.join('cifar10_dataset', 'noisy_labels_{}'.format(seed))
    if not os.path.isdir(data_path_noisy):
        os.mkdir(data_path_noisy)

    print('Generating label noise with seed {:2d}...'.format(seed))

    for noise in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        for sparsity in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            if (noise != 0.0) or ((noise == 0.0) and (sparsity == 0)):

                # Generate class-conditional noise
                noise_matrix = noise_generation.generate_noise_matrix_from_trace(
                    K=K, trace=int(K * (1 - noise)),
                    valid_noise_matrix=False, frac_zero_noise_rates=sparsity,
                    seed=seed)

                # Create noisy labels
                np.random.seed(seed=seed)
                y_noisy = noise_generation.generate_noisy_labels(y_true, noise_matrix)

                # Create map of filenames to noisy labels
                d = dict(zip([i for i, j in train_dataset.imgs], [int(i) for i in y_noisy]))

                # Store noisy labels as json
                wfn_base = 'noisylabels_noise{}_sparsity{}.json'.format(noise, sparsity)
                wfn = os.path.join(data_path_noisy, wfn_base)

                with open(wfn, 'w') as wf:
                    wf.write(json.dumps(d))

                # Store noise matrix as json
                wfn_base = 'noisematrix_noise{}_sparsity{}.pickle'.format(noise, sparsity)
                wfn = os.path.join(data_path_noisy, wfn_base)

                with open(wfn, 'wb') as wf:
                    pickle.dump(noise_matrix, wf, protocol=pickle.HIGHEST_PROTOCOL)
