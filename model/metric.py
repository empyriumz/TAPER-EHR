import torch

from sklearn.metrics import (
    roc_curve,
    auc,
    roc_curve,
    precision_recall_curve,
)


def roc_auc(output, target):
    # temporary place holders..
    # these will be run at the end of the epoch once all probabilities are obtained..
    # refer to pr_auc_1

    # output = output.detach().cpu().numpy()
    # target = target.cpu().numpy()
    # if (torch.sum(target) == 0 or torch.sum(target) == len(target)):
    #    return 1.0

    # return roc_auc_score(target, output)
    return 1.0


def pr_auc(output, target):
    # temporary place holders.
    # these will be run at the end of the epoch once all probabilities are obtained..
    # refer to pr_auc_1

    # output = output.detach().cpu().numpy()
    # target = target.cpu().numpy()
    # if (torch.sum(target) == 0 or torch.sum(target) == len(target)):
    #    return 1.0
    # return average_precision_score(target, output)
    return 1.0


def roc_auc_1(output, target):
    # evaluate after eac
    fpr, tpr, thresholds = roc_curve(target, output)
    area = auc(fpr, tpr)
    return area  # roc_auc_score(target, output)


def pr_auc_1(output, target):
    precision, recall, _ = precision_recall_curve(target, output)
    area = auc(recall, precision)
    return area  # average_precision_score(target, output)


def accuracy(output, target):
    with torch.no_grad():
        pred = output >= 0.5  # torch.argmax(output, dim=1)
        pred = pred.long()
        correct = 0
        correct += torch.sum(pred == target).item()

    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def recall(output, target, mask, k=10, window=1):
    bsz = output.shape[0]
    idx = torch.arange(0, bsz, device=output.device)

    mask = mask.squeeze()
    for i in range(window):
        mi = mask[i + 1 :] * mask[: -i - 1]
        mi = torch.nn.functional.pad(mi, (1 + i, 1 + i))
        tm = mi[: -i - 1]
        im = mi[i + 1 :]

        target_mask = torch.masked_select(idx, tm)
        input_mask = torch.masked_select(idx, im)
        output = output[input_mask, :]
        output = output.float()
        target = target[target_mask, :]
        target = target.float()

        _, tk = torch.topk(output, k)
        tt = torch.gather(target, 1, tk)
        r = torch.mean(torch.sum(tt, dim=1) / (torch.sum(target, dim=1) + 1e-7))
        if r != r:
            r = 0
    return r

def recall_10(output, target, mask, k=20, window=1):
    return recall(output, target, mask, k=10, window=1)

def recall_20(output, target, mask, k=20, window=1):
    return recall(output, target, mask, k=20, window=1)

def recall_30(output, target, mask, k=30, window=1):
    return recall(output, target, mask, k=30, window=1)

def recall_40(output, target, mask, k=40, window=1):
    return recall(output, target, mask, k=40, window=1)

def recall_50(output, target, mask, k=50, window=1):
    return recall(output, target, mask, k=50, window=1)

def specificity(output, target, t=0.5):
    with torch.no_grad():
        preds = output > t  # torch.argmax(output, dim=1)
        preds = preds.long()
        num_true_0s = torch.sum(
            (target == 0) & (preds == target), dtype=torch.float
        ).item()
        num_false_1s = torch.sum(
            (target == 0) & (preds != target), dtype=torch.float
        ).item()

    if num_false_1s == 0:
        return 1
    s = num_true_0s / (num_true_0s + num_false_1s)

    if s != s:
        s = 1

    return s

def sensitivity(output, target, t=0.5):
    with torch.no_grad():
        preds = output > t  # torch.argmax(output, dim=1)
        preds = preds.long()
        num_true_1s = torch.sum((preds == target) & (preds == 1), dtype=torch.float)
        num_false_1s = torch.sum((preds != target) & (preds == 1), dtype=torch.float)

    s = num_true_1s / (num_true_1s + num_false_1s)

    if s != s:
        s = 1
    return s


def precision(output, target, t=0.5):
    with torch.no_grad():
        preds = output > t
        preds = preds.long()
        num_true_1s = torch.sum((preds == target) & (preds == 1), dtype=torch.float)
        num_false_0s = torch.sum((preds != target) & (preds == 0), dtype=torch.float)

    s = num_true_1s / (num_true_1s + num_false_0s)
    if s != s:
        s = 1
    return s
