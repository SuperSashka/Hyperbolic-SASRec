import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

tensor_dir = 'saved_tensors'
pattern = re.compile(r'^(p_aff|p_dist|n_aff|n_dist)_(\d+)_\d+\.pt$')

# collect tensors by type and epoch
data = defaultdict(lambda: defaultdict(list))
for fname in os.listdir(tensor_dir):
    m = pattern.match(fname)
    if not m:
        continue
    ttype, epoch_str = m.groups()
    epoch = int(epoch_str)
    tensor = torch.load(os.path.join(tensor_dir, fname))
    vals = tensor.detach().cpu().flatten().numpy()
    # apply filters
    if ttype.endswith('_dist'):
        vals = vals[vals > 0]
    else:  # affinity
        vals = np.log(vals[vals!=0])
    if vals.size > 0:
        data[ttype][epoch].append(vals)

# for each tensor type, build per-epoch arrays, print stats and boxplot
for ttype in ('p_aff','p_dist','n_aff','n_dist'):
    epochs = sorted(data[ttype].keys())
    if not epochs:
        print(f"No data for {ttype}")
        continue
    box_data = []
    print(f"\n{ttype} statistics by epoch (filtered):")
    for e in epochs:
        arr = np.concatenate(data[ttype][e])
        box_data.append(arr)
        print(f"  epoch {e:2d}: N={arr.size}  mean={arr.mean():.4f}  std={arr.std():.4f}")
    # plot
    plt.figure(figsize=(8,4))
    plt.boxplot(box_data,
                tick_labels=[str(e) for e in epochs],
                showfliers=False)
    plt.title(f"{ttype} per epoch")
    plt.xlabel("epoch")
    plt.ylabel(ttype)
    if ttype.endswith('aff'):
        plt.ylim(bottom=-10)
    plt.grid(True)
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(tensor_dir, f'{ttype}_boxplots.png'))