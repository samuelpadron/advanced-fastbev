import json
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime

def visualize(config: str) -> None:
    log_files = sorted(glob.glob(f'work_dirs/fastbev-temporal-r50-cbgs{config}/*.log.json', recursive=True))

    timestamp = datetime.datetime.now().strftime('%b-%d-%X')
    save_name = f"viz_output/loss_curve-{timestamp}.png"

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Training Plots',
        description='Visualize training losses'
    )
    parser.add_argument('config')
    args = parser.parse_args()
    visualize(args.config)