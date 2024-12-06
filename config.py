# File Path #
log_file = './stdout.log'
train_file = "./yelp2018/train.txt"
test_file = "./yelp2018/test.txt"
adj_mat_file = "./yelp2018/s_pre_adj_mat.npz"
metric_result_file = './result/metric.tsv'
best_model_dir = './result/best_model'

# Hyperparams #
epochs = 100
n_layers = 3
embed_dim = 64
train_batch_size = 2048
test_batch_size = 1024
do_neg_sampling = True
reg_strength = 1e-4

# Loss #
loss_fn = 'bpr'

# Eval #
topk = 20
metrics = ['recall', 'ndcg', 'diversity']

# Random Seed #
seed = 2024