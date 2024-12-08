import pandas as pd
import matplotlib.pyplot as plt


def twin_plot(df, title, out):
    fig, ax1 = plt.subplots(figsize=(7, 6), dpi=1000)

    ax1.plot(df['epoch'], df['diversity'], label='diversity', color='orangered')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Diversity', color='black')

    ax2 = ax1.twinx()  
    ax2.plot(df['epoch'], df['recall@20'], label='recall@20', color='dodgerblue')
    ax2.plot(df['epoch'], df['ndcg@20'], label='ndcg@20', color='limegreen')
    ax2.set_ylabel('Recall@20 & NDCG@20', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(handles + handles2, labels + labels2, loc='lower right')

    plt.title(title)

    plt.grid(False)
    plt.savefig(out)
    plt.close()


base = pd.read_csv('./result/lightgcn-reproduce/metric.tsv', sep='\t')
ours = pd.read_csv('./result/deweighted-loss/metric.tsv', sep='\t')

twin_plot(base, 'Figure2-(a): LightGCN-BPR', out='./figures/fig2(a)-LightGCN-BPR.png')
twin_plot(ours, 'Figure2-(b): Ours', out='./figures/fig2(b)-Ours.png')