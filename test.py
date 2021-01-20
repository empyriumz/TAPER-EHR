import os
import argparse
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model.metric import roc_auc_1, pr_auc_1
from train import get_instance
from train import import_module
import json
import pickle


def main(test_config):
    # load model architecture
    model_path = test_config["trained_model_path"]
    model_config = torch.load(model_path)["config"]   
    model = import_module("model", model_config)(**model_config["model"]["args"])
    model.summary()
    
    # setup data_loader instances
    data_loader = get_instance(module_data, "data_loader", test_config)
    weight = data_loader.dataset.get_pos_weight()
    print(weight)
    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, model_config["loss"])
    metric_fns = [getattr(module_metric, met) for met in model_config["metrics"]]

    # load state dict
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]
    if model_config["n_gpu"] > 1:
        model = torch.nn.DaaParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = np.zeros(len(metric_fns))
    predictions = {"output": [], "target": []}

    with torch.no_grad():
        for data, target in data_loader:
            target = target.to(device)
            if len(target.shape) == 0:
                target = target.unsqueeze(dim=0)
            output = model(data, device)
            if model_config["loss"] == "bce_loss":
                output, _ = model(data, device=device)
            elif model_config["loss"] == "bce_loss_with_logits":
                _ ,output= model(data, device=device)
            predictions["output"].append(output.cpu().numpy())
            predictions["target"].append(target.cpu().numpy())

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = target.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
            del output
            del target
            
    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update(
        {
            met.__name__: total_metrics[i].item() / n_samples
            for i, met in enumerate(metric_fns)
        }
    )
    predictions["output"] = np.hstack(predictions["output"])
    predictions["target"] = np.hstack(predictions["target"])
    print(len(data_loader.dataset), n_samples, predictions["output"].shape, predictions["target"].shape)
    total_metrics[-2] = pr_auc_1(predictions["output"], predictions["target"])
    total_metrics[-1] = roc_auc_1(predictions["output"], predictions["target"])
    log.update({metric_fns[-2].__name__: total_metrics[-2]})
    log.update({metric_fns[-1].__name__: total_metrics[-1]})
    print(log)
    save_dir = os.path.join(os.path.abspath(os.path.join(model_path, "..")))

    with open(os.path.join(save_dir, "predictions.pkl"), "wb") as handle:
        pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(os.path.join(save_dir, "test-results.pkl"), "wb") as handle:
        pickle.dump(log, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")

    parser.add_argument(
        "-t",
        "--test",
        default=None,
        type=str,
        help="test dataloader config",
    )

    args = parser.parse_args()
    assert args.test != None, "need data and model to test"
    test_config = json.load(open(args.test))
   
    main(test_config)
