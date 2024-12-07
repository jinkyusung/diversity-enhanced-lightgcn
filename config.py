import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# File Path #
log_file = "./stdout.log"
train_file = "./yelp2018/train.txt"
test_file = "./yelp2018/test.txt"
adj_mat_file = "./yelp2018/s_pre_adj_mat.npz"
metric_result_file = f"./result/{current_time}_metric.tsv"
best_model_dir = f"./result/{current_time}_best_model"

load_model = False
if load_model:
    model_file = "./best_model_epoch1.pth"

# Hyperparams #
start_epoch = 1  # should be 1 typically, and not 1 if you want to continue training
epochs = 1000  # meaning the end epoch, so `epochs - start_epoch + 1` is the number of epochs to train
n_layers = 3
embed_dim = 64
train_batch_size = 2048
test_batch_size = 1024
do_neg_sampling = True
reg_strength = 1e-4

# Loss #
loss_fn = "bpr"

# Eval #
topk = 20
metrics = ["recall", "ndcg", "diversity"]

# Random Seed #
seed = 2024

# Sanity Check #
assert start_epoch <= epochs, "start_epoch should be less than or equal to epochs"
assert (load_model == True and start_epoch > 1) or (load_model == False and start_epoch == 1), "start_epoch should be 1 if load_model is False"
assert (loss_fn == "bpr" and do_neg_sampling == True) or (loss_fn != "bpr"), "BPR loss requires negative sampling"