import json
import glob
import numpy as np
import matplotlib.pyplot as plt


log_files = sorted(glob.glob('work_dirs/fastbev-temporal-r50-cbgs/**/*.log.json', recursive=True))

save_name = "viz_output/loss_curve_normal_weight.png"

print(f'Found {len(log_files)} log files:')

iters, loss, heatmap, xy = [], [], [], []
for log_file in log_files:
    with open(log_file) as f:
        for line in f:
            log = json.loads(line.strip())
            if log.get('mode') == 'train':
                global_iter = (log['epoch'] - 1) * 1000 + log['iter']
                iters.append(global_iter)
                loss.append(log['loss'])
                heatmap.append(log['task0.loss_heatmap'])
                xy.append(log['task0.loss_xy'])

# Sort everything by iter in case logs overlap
combined = sorted(zip(iters, loss, heatmap, xy))
iters, loss, heatmap, xy = zip(*combined)
iters = np.array(iters)
print(f'Total data points: {len(iters)}')
print(f'Iter range: {iters[0]} to {iters[-1]}')


def smooth(y, w=20):
    return np.convolve(y, np.ones(w)/w, mode='valid')


plt.style.use('dark_background')
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

for ax, values, label, color in zip(
    axes,
    [loss, heatmap, xy],
    ['Total Loss', 'Heatmap Loss', 'XY Loss'],
    ['#4C9BE8', '#E8834C', '#4CE896']
):
    raw = np.array(values)
    sm  = smooth(raw)
    ax.plot(iters, raw, color=color, linewidth=0.5, alpha=0.3)
    ax.plot(iters[:len(sm)], sm, color=color, linewidth=2.0, label='smoothed')
    ax.set_title(label)
    ax.set_xlabel('iter')
    ax.legend()

plt.suptitle('FastBEV Training Loss (all runs)', fontsize=14)
plt.tight_layout()
plt.savefig(save_name, dpi=150)

print(f'Saved {save_name}')