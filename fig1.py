import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./result/lightgcn-reproduce/metric.tsv', sep='\t')

x = df['epoch']
y_recall = df['recall@20']
y_ndcg = df['ndcg@20']
y_diversity = df['diversity']


plt.figure(figsize=(6, 6), dpi=1000)

plt.plot(x, y_recall, label='recall@20', color='dodgerblue')
plt.plot(x, y_ndcg, label='ndcg@20', color='limegreen')

plt.axhline(0.0649, label='recall@20 in paper', color='black', linestyle='-.', linewidth=1)
plt.axhline(0.0530, label='ndcg@20 in paper', color='black', linestyle='--', linewidth=1)

plt.xlabel('Epoch', fontsize=13)

plt.legend()
plt.title('Figure1: LightGCN-BPR Reproduce Sanity Check', fontsize=14)

plt.savefig('./figures/fig1-reproduce.png')
plt.grid(False)

plt.close()