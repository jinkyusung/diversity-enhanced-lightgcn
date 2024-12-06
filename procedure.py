import torch
import yaml
from evaluator import TopKEvaluator
from data_utils import remove_padding


# Variables for config #
class Config:
    def __init__(self, data):
        for key, value in data.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)

with open("config.yml", "r") as file:
    data = yaml.safe_load(file)
    c = Config(data)


# File output function for logging #
def console(arg) -> None:
    msg = str(arg) + '\n'
    with open(c.path.log, 'a') as fp:
        fp.write(msg)
    return


# Procedure #
def train_loop(train_dataloader, model, loss_fn, optimizer: torch.optim.Optimizer, device):
    model.train()

    loss_sum = 0

    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)

    for batch_num, minibatch in enumerate(train_dataloader):
        optimizer.zero_grad()

        user:           torch.Tensor = minibatch[0].to(device)
        pos_item:       torch.Tensor = minibatch[1].to(device)
        if c.model.neg_sampling:
            neg_item:   torch.Tensor = minibatch[2].to(device)

        if c.model.neg_sampling:
            result = model(user, pos_item, neg_items=neg_item)
        else:
            result = model(user, pos_item)

        loss = loss_fn(**result, pos_item=pos_item)
        
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

        if batch_num % 100 == 0:
            console(
                f"loss: {loss.item():>7f} [{c.batch.train * batch_num + len(minibatch[0]):>5d}/{size:>5d}]"
            )


def test_loop(dataloader, model, loss_fn, evaluator: TopKEvaluator, epoch: int, device):
    global best_metric, write_header

    model.eval()

    num_batches = len(dataloader)

    metrics_result_dict = dict()

    with torch.no_grad():
        for minibatch in dataloader:
            user:       torch.Tensor = minibatch[0].to(device)
            history:    torch.Tensor = minibatch[1].to(device)  # 각 유저 별 train 에서 존재하는 아이템
            label:      torch.Tensor = minibatch[2].to(device)  # 각 유저 별 test  에서 존재하는 아이템

            history:    list[torch.Tensor] = remove_padding(history)
            label:      list[torch.Tensor] = remove_padding(label)

            pred:       torch.Tensor = model.get_users_rating_prediction(user)

            result_dict = evaluator.evaluate(pred, history, label)
            for metric in result_dict:
                if metric not in metrics_result_dict:
                    metrics_result_dict[metric] = 0
                metrics_result_dict[metric] += result_dict[metric]

    for metric in metrics_result_dict:
        metrics_result_dict[metric] /= num_batches

    # Save metrics to a text file
    with open(c.path.result, "a") as f:
        if write_header:
            f.write("epoch\t")
            for metric in metrics_result_dict:
                f.write(f"{metric}\t")
            f.write("\n")
            write_header = False

        f.write(f"{epoch}\t")
        for metric in metrics_result_dict:
            f.write(f"{metrics_result_dict[metric]:.4f}\t")
        f.write("\n")

    # Check and save the best models
    for metric in metrics_result_dict:
        if metric not in best_metric:
            best_metric[metric] = 0
        if metrics_result_dict[metric] > best_metric[metric]:
            best_metric[metric] = metrics_result_dict[metric]
            console(f"Best {metric} model updated. Saving the model.")
            torch.save(
                model.state_dict(), f"{c.path.model}/best_{metric}_model.pth"
            )

    console(f"Eval results: ")
    for metric in metrics_result_dict:
        console(f"{metric}: {metrics_result_dict[metric]:.4f}  ")
    console("\n")