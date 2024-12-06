import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./deweighted-loss/metric.tsv', sep='\t')

x = df['epoch']
y_recall = df['recall@20']
y_ndcg = df['ndcg@20']
y_diversity = df['diversity']

fig, ax1 = plt.subplots(figsize=(7, 6), dpi=1000)

ax1.plot(x, y_diversity, label='diversity', color='orangered')

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Diversity', color='black')

ax2 = ax1.twinx()  
ax2.plot(x, y_recall, label='recall@20', color='dodgerblue')
ax2.plot(x, y_ndcg, label='ndcg@20', color='limegreen')
ax2.set_ylabel('Recall@20 & NDCG@20', color='black')
ax2.tick_params(axis='y', labelcolor='black')

ax1.axhline(0.1600, color='lightgray', linestyle='--', linewidth=2)
ax2.axhline(0.0649, color='lightgray', linestyle='--', linewidth=2)
ax2.axhline(0.0530, color='lightgray', linestyle='--', linewidth=2)

handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(handles + handles2, labels + labels2, loc='lower right')

plt.title('Metric Comparison over Epochs')

plt.grid(False)
plt.savefig('./figure1.png')
plt.close()