import operator

import numpy as np
import torch

from base import BaseTrainer
from utils import operator_on_dict, median_scaling


class Evaluater(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.trainer
    """
    def __init__(self, model, loss, metrics, config, data_loader):
        '''
        # 传入了模型, loss, 需要记录的metrics, config, 测试数据
        '''
        super().__init__(model, loss, metrics, None, config)
        self.config = config
        self.data_loader = data_loader
        self.log_step = config["evaluater"].get("log_step", int(np.sqrt(data_loader.batch_size)))
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.len_data = len(self.data_loader)

        if isinstance(loss, torch.nn.Module):
            self.loss.to(self.device)
            if len(self.device_ids) > 1:
                self.loss = torch.nn.DataParallel(self.loss, self.device_ids)
        
        # 没有set就设置为None
        self.roi = config["evaluater"].get("roi", None)
        self.alpha = config["evaluater"].get("alpha", None)
        self.max_distance = config["evaluater"].get("max_distance", None)  # 80
        self.correct_length = config["evaluater"].get("correct_length", False)
        self.median_scaling = config["evaluater"].get("median_scaling", False)

    def _eval_metrics(self, data_dict):
        acc_metrics = np.zeros(len(self.metrics))  # (7,) - 7个0
        for i, metric in enumerate(self.metrics):  # 对于每个metrics都for一次
            if self.median_scaling:
                data_dict = median_scaling(data_dict)
            # 运行function, abs_rel_sparse_metrics
            acc_metrics[i] += metric(data_dict, self.roi, self.max_distance)
            #self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        if np.any(np.isnan(acc_metrics)):
            acc_metrics = np.zeros(len(self.metrics))
            valid = np.zeros(len(self.metrics))
        else:
            valid = np.ones(len(self.metrics))
        return acc_metrics, valid

    def eval(self, model_index):
        """
        Training logic for an epoch

        :param model_index: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.eval()  # 设置为eval

        total_loss = 0   # 初始化 loss = 0
        total_loss_dict = {}
        # 1.初始化7个metrics的值
        total_metrics = np.zeros(len(self.metrics))   # (7,) - 7个0
        total_metrics_valid = np.zeros(len(self.metrics))  # (7,) - 7个0

        total_metrics_runningavg = np.zeros(len(self.metrics))
        num_samples = 0

        for batch_idx, (data, target) in enumerate(self.data_loader):
            # 读取数据
            data, target = to(data, self.device), to(target, self.device)  # data-dict: keyframe-[B,3,256,512],keyframe_pose-[B,4,4]
            data["target"] = target  # 把target增加到data dict中去

            with torch.no_grad():
                data = self.model(data)
                # loss 设置为全0
                loss_dict = {"loss": torch.tensor([0])}   # 后续的也是0
                loss = loss_dict["loss"]
            
            # result - tensor [B,1,256,512] - 最终的depth map image
            output = data["result"]  # 测试一个，是0.1896～0.0025，即5.27m～400m

            #self.writer.set_step((model_index - 1) * self.len_data + batch_idx)
            #self.writer.add_scalar('loss', loss.item())
            ## 把loss与之前的进行求和
            total_loss += loss.item()
            total_loss_dict = operator_on_dict(total_loss_dict, loss_dict, operator.add)   # 每个metrics的数都相加
            ## 求这次data的metrics指标
            metrics, valid = self._eval_metrics(data)  # metrics是[7个metric，如果不valid，则置0]，valid代表[每个metric处是否valid]
            total_metrics += metrics  ## 求和
            total_metrics_valid += valid

            batch_size = target.shape[0]
            if num_samples == 0:
                total_metrics_runningavg += metrics
            else:
                total_metrics_runningavg = total_metrics_runningavg * (num_samples / (num_samples + batch_size)) + \
                                           metrics * (batch_size / (num_samples + batch_size))
            num_samples += batch_size

            if batch_idx % self.log_step == 0:  # 每20个batch打印一次
                # 形式为: 20%进度，loss指标，平均metrics指标
                self.logger.debug(f'111Evaluating {self._progress(batch_idx)} Loss: {loss.item() / (batch_idx + 1):.6f} Metrics: {list(total_metrics / (batch_idx + 1))}')
                self.logger.debug(f'222Evaluating {self._progress(batch_idx)} Loss: {loss.item() / (batch_idx + 1):.6f} Metrics: {list(total_metrics / total_metrics_valid)}')
                # np.array(list((map('{:.6f}%'.format,total_metrics_valid))))
                #self.writer.add_image('input', make_grid(to(data["keyframe"], "cpu"), nrow=3, normalize=True))
                #self.writer.add_image('output', make_grid(to(torch.clamp(1 / output, 0, 100), "cpu") , nrow=3, normalize=True))
                #self.writer.add_image('ground_truth', make_grid(to(torch.clamp(1 / target, 0, 100), "cpu"), nrow=3, normalize=True))

            if batch_idx == self.len_data:
                break

        log = {
            'loss': total_loss / self.len_data,   # loss = 0
            'metrics': (total_metrics / total_metrics_valid).tolist(),    # 本身的metrics, length=7
            'metrics_correct': total_metrics_runningavg.tolist(),  # 调整后的metrics, length=7
            'valid_batches': total_metrics_valid[0]    # 有效的batches个数
        }
        for loss_component, v in total_loss_dict.items():
            # 最后求一个平均值，返回
            log[f"loss_{loss_component}"] = v.item() / self.len_data

        return log

    def _progress(self, batch_idx):
        # 获取当前百分比
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_data
        return base.format(current, total, 100.0 * current / total)


def to(data, device):
    if isinstance(data, dict):
        return {k: to(data[k], device) for k in data.keys()}
    elif isinstance(data, list):
        return [to(v, device) for v in data]
    else:
        return data.to(device)
