import pandas as pd
import matplotlib.pyplot as plt


bpr = pd.read_csv('./result/bpr/metric.tsv', sep='\t')
dw_bpr = pd.read_csv('./result/dw-bpr/metric.tsv', sep='\t')
directau = pd.read_csv('./result/directau/metric.tsv', sep='\t')
ours = pd.read_csv('./result/ours/metric.tsv', sep='\t')


def cmp_plot(loss_list, label_list, metric, palette):
    plt.figure(figsize=(6, 6), dpi=1000)

    for i, loss in enumerate(loss_list):
        x_axis = loss['epoch']
        y_axis = loss[metric]

        if len(x_axis) > 80:
            x_axis = x_axis[:80]
            y_axis = y_axis[:80]

        color = palette[i]
        label = label_list[i]
        plt.plot(x_axis, y_axis, label=label, color=color)

    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel(metric, fontsize=13)
    plt.legend()
    plt.title(f'Figure2: Training Curve ({metric})', fontsize=14)
    plt.savefig(f'./figures/fig2-comparision-{metric}.png')
    plt.grid(False)
    plt.close()


loss_list = [bpr, dw_bpr, directau, ours]
label_list = ['BPR', 'Deweighted-BPR', 'DirectAU', 'Ours']
palette = ['orangered', 'gold', 'limegreen', 'dodgerblue']
cmp_plot(loss_list, label_list, 'recall@20', palette)
cmp_plot(loss_list, label_list, 'ndcg@20', palette)
cmp_plot(loss_list, label_list, 'diversity', palette)
