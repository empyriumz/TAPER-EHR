import numpy as np
import torch
from base import BaseTrainer
from model.metric import roc_auc_1, pr_auc_1, pr_auc, roc_auc


class ClassificationTrainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
    """

    def __init__(
        self,
        model,
        loss,
        metrics,
        optimizer,
        resume,
        config,
        data_loader,
        valid_data_loader=None,
        lr_scheduler=None,
        train_logger=None,
    ):
        super(ClassificationTrainer, self).__init__(
            model, loss, metrics, optimizer, resume, config, train_logger
        )
        self.config = config
        self.data_loader = data_loader
        self.pos_weight = self.data_loader.pos_weight
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config["trainer"].get(
            "log_step", int(np.sqrt(data_loader.batch_size))
        )

        if self.config["model"]["args"]["num_classes"] == 1:
            weight_0 = self.config["trainer"].get("class_weight_0", 1.0)
            weight_1 = self.config["trainer"].get("class_weight_1", 1.0)
            self.weight = [weight_0, weight_1]
            if self.config["loss"] == "bce_loss":
                self.loss = lambda output, target: loss(output, target, self.weight)
            elif self.config["loss"] == "bce_loss_with_logits":
                self.loss = lambda output, target: loss(output, target, weight=self.weight, 
                                                        pos_weight = self.pos_weight)
        self.prauc_flag = pr_auc in self.metrics and roc_auc in self.metrics

    def _eval_metrics(self, output, target, **kwargs):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(output, target, **kwargs)
            self.writer.add_scalar(f"{metric.__name__}", acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        all_t = []
        all_o = []
        for batch_idx, (data, target) in enumerate(self.data_loader):
            all_t.append(target.numpy())
            target = target.to(self.device)
            self.optimizer.zero_grad()
            if self.config["loss"] == "bce_loss":
                output, _ = self.model(data, device=self.device)
            elif self.config["loss"] == "bce_loss_with_logits":
                _ ,output= self.model(data, device=self.device)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * len(self.data_loader) + batch_idx)
            self.writer.add_scalar("loss", loss.item())

            total_loss += loss
            total_metrics += self._eval_metrics(output, target)
            all_o.append(output.detach().cpu().numpy())

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] {}: {:.6f}".format(
                        epoch,
                        batch_idx * self.data_loader.batch_size,
                        self.data_loader.n_samples,
                        100.0 * batch_idx / len(self.data_loader),
                        "loss",
                        loss,
                    )
                )

        total_metrics = total_metrics / len(self.data_loader)
        if self.prauc_flag:
            all_o = np.hstack(all_o)
            all_t = np.hstack(all_t)
            total_metrics[-2] = pr_auc_1(all_o, all_t)
            total_metrics[-1] = roc_auc_1(all_o, all_t)

        log = {
            "loss": total_loss / len(self.data_loader),
            "metrics": total_metrics,
        }

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log = {**log, **val_log}

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        all_t = []
        all_o = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                all_t.append(target.numpy())
                target = target.to(self.device)

                output, _ = self.model(data, self.device)
                loss = self.loss(
                    output,
                    target.reshape(
                        -1,
                    ),
                )
                all_o.append(output.detach().cpu().numpy())

                self.writer.set_step(
                    (epoch - 1) * len(self.valid_data_loader) + batch_idx, "valid"
                )
                self.writer.add_scalar("loss", loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(output, target)

        total_val_metrics = (total_val_metrics / len(self.valid_data_loader)).tolist()
        if self.prauc_flag:
            all_o = np.hstack(all_o)
            all_t = np.hstack(all_t)
            total_val_metrics[-2] = pr_auc_1(all_o, all_t)
            total_val_metrics[-1] = roc_auc_1(all_o, all_t)

        return {
            "val_loss": total_val_loss / len(self.valid_data_loader),
            "val_metrics": total_val_metrics,
        }
