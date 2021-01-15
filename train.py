import os
import json
import argparse
import torch
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from utils import Logger
from model import optimization

def get_instance(module, name, config, *args):
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])

def import_module(name, config):
    return getattr(
        __import__("{}.{}".format(name, config[name]["module_name"])),
        config[name]["type"],
    )

def main(config, resume, test):
    train_logger = Logger()

    # setup data_loader instances
    data_loader = get_instance(module_data, "data_loader", config)
    valid_data_loader = None
    if test == 0:
        assert data_loader.test == False, "incompatible configs, please set test to false"
        valid_data_loader = data_loader.split_validation()       
      
    # build model architecture
    model = import_module("model", config)(**config["model"]["args"])
    # model = get_instance(module_arch, 'arch', config)
    print(model)

    # get function handles of loss and metrics
    loss = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, "optimizer", config, trainable_params)
    try:
        lr_scheduler = get_instance(
            optimization, "lr_scheduler", config, optimizer
        )
    except:
        lr_scheduler = get_instance(
            torch.optim.lr_scheduler, "lr_scheduler", config, optimizer
        )
    Trainer = import_module("trainer", config)
    trainer = Trainer(
        model,
        loss,
        metrics,
        optimizer,
        resume=resume,
        config=config,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
        train_logger=train_logger,
    )
    if test == 0:
        trainer.train()
    else:
        trainer.test()
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structmed Trainer")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-t",
        "--test",
        default=0,
        type=int,
        help="enable test mode",
    )
    args = parser.parse_args()
    assert args.test in [0, 1], "invalid test mode!"
    if args.config:
        # load config file
        config = json.load(open(args.config))
        path = os.path.join(config["trainer"]["save_dir"], config["name"])
    elif args.resume:
        # load config file from checkpoint, in case new config file is not given.
        # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
        config = torch.load(args.resume)["config"]
    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.json', for example."
        )

    main(config, args.resume, args.test)
