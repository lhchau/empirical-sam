#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import argparse
import importlib
import os
import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from .utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal
from .hessian import hessian

def get_eigen_hessian_plot(name, net, criterion, dataloader, hessian_batch_size=5120, mini_hessian_batch_size=128):
    save_path = f'checkpoint/{name}/ckpt_best.pth'

    checkpoint = torch.load(save_path)
    net.load_state_dict(checkpoint['net'])
    net.eval()
    
    batch_num = hessian_batch_size // mini_hessian_batch_size
    
    if batch_num == 1:
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            hessian_dataloader = (inputs, labels)
            break
    else:
        hessian_dataloader = []
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.cuda(), labels.cuda()
            hessian_dataloader.append((inputs, labels))
            if i == batch_num - 1:
                break
    
    if batch_num == 1:
        hessian_comp = hessian(net,
                            criterion,
                            data=hessian_dataloader,
                            cuda=True)
    else:
        top_eigenvalues_list = []
        for hessian_data in tqdm(hessian_dataloader):
            hessian_comp = hessian(net,
                                criterion,
                                data=hessian_data,
                                cuda=True)
            top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=5)
            top_eigenvalues_list.extend(top_eigenvalues)

    data = top_eigenvalues_list

    sns.kdeplot(data, fill=True)

    positive_values = [x for x in data if x > 0]
    max_positive = np.max(positive_values)
    expectation_positive = np.mean(positive_values)

    # Calculate probability of negative values
    negative_values = [x for x in data if x < 0]
    probability_negative = len(negative_values) * 100 / len(data)
    max_negative_values = min(negative_values) if len(negative_values) else 0

    # Add labels and title
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density')
    plt.title('Density Plot')

    # Annotate the plot with expectation and probability information
    plt.axvline(expectation_positive, color='red', linestyle='dashed', linewidth=2, label=f'Expectation: {expectation_positive:.2f}')
    plt.figtext(0.55, 0.65,f'Max Pos: {max_positive:.0f}')
    plt.figtext(0.55, 0.55,f'% Neg: {probability_negative:.3f}%, Max Neg: {max_negative_values:.0f}')
    plt.legend()

    save_plot_dir = f'./density_plot/{name}'
    if not os.path.exists(save_plot_dir):
        os.makedirs(save_plot_dir)
    # plt.savefig(f'{save_plot_dir}/ckpt_best.png', dpi=300)
    # Show the plot
    # plt.close()
    return plt
