# append root
import os
import sys
a = (os.path.abspath(sys.argv[0])[0: os.path.abspath(sys.argv[0]).find('sparse_sharing')])
b = 'sparse_sharing'
root = os.path.join(a, b)
sys.path.append(root)
from fastai.vision.all import Callback, TrackerCallback, store_attr, ReduceLROnPlateau, EarlyStoppingCallback, CancelFitException
import numpy as np
import torch
from torch import masked_fill

class ChooseTask(Callback):
    """
    选择一个任务做dataloader
    """
    def __init__(self, task=1):
        """
        :param task: 选择任务 1/2
        """
        super(ChooseTask, self).__init__()
        self.task = task

    def before_fit(self):
        self.learn.loss_func = self.loss_func[self.task-1]

    def before_batch(self):
        self.learn.yb = (self.yb[self.task-1], )


class SaveModelCallback1(TrackerCallback):
    """
    改进SaveModelCallback
    """
    def __init__(self, monitor='valid_loss', comp=None, min_delta=0., every_epoch=False, with_opt=False):
        """
        :param monitor: 评价指标
        :param min_delta: 评价指标的最小差距
        :param every_epoch: 是否每一个epoch都保存
        :param with_opt: 保存时候是否将优化参数保存
        """
        super().__init__(monitor=monitor, comp=comp, min_delta=min_delta)
        self.last_saved_path = None
        store_attr('every_epoch,with_opt')

    def before_train(self):
        self._save(os.path.join(root, 'ckpt', f'{self.learn.model.name}_bs{self.learn.dls.bs}_origin'))

    def _save(self, name):
        self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)

    def after_epoch(self):
        # "Compare the value monitored to its best score and save if best."
        if self.every_epoch:
            self.fname = os.path.join(root, 'ckpt', f'{self.learn.model.name}_bs{self.learn.dls.bs}_v{self.best}')
            self._save(self.fname)
        else:  # every improvement
            super().after_epoch()
            self.fname = os.path.join(root, 'ckpt', f'{self.learn.model.name}_bs{self.learn.dls.bs}_v{self.best}')
            if self.new_best:
                print(f'Better model found at epoch {self.epoch} with {self.monitor} value: {self.best}.')
                self._save(self.fname)

    def after_fit(self, **kwargs):
        if not self.every_epoch: self.learn.load(f'{self.fname}')
        try:
            os.rmdir(os.path.join(self.learn.path, 'models'))
        except:
            print(f"Cannot delete {os.path.join(self.learn.path, 'models')}!")


class LoadCkpt(Callback):
    def __init__(self, ckpt_name=None, every_batch=False):
        super(LoadCkpt, self).__init__()
        self.ckpt_path = os.path.join(root, 'ckpt', ckpt_name)
        if self.ckpt_path.find('.pt') != -1:
            self.ckpt_path = self.ckpt_path[: self.ckpt_path.find(".pt")]
        self.every_batch = every_batch

    def before_fit(self):
        try:
            print(f"Loading {self.ckpt_path}...")
            self.learn.load(self.ckpt_path)
        except:
            print(f"Cannot load {self.ckpt_path}! Check if it exist!")

    def before_batch(self):
        if self.every_batch:
            self.before_fit()
        else:
            pass


class LoadMask(Callback):
    def __init__(self, mask_name=None, every_batch=False):
        super(LoadMask, self).__init__()
        assert mask_name, "Please check your mask path."
        self.mask_path = os.path.join(root, 'ckpt', mask_name)
        self.masks = torch.load(self.mask_path)
        self.every_batch = every_batch

    def before_fit(self):
        print("Start loading mask!")
        self.param_list = self.learn.opt.param_groups[0]['params'].items
        masked_param_list = []
        for ind, param in enumerate(self.param_list):
            print(f"Layer {ind}: ")
            print(f"Original param is {np.count_nonzero(param.data.cpu().numpy())}")
            mask = self.masks[ind]
            mask_tensor = torch.tensor(mask).cuda() if torch.cuda.is_available() else torch.tensor(mask)
            masked_param = torch.mul(mask_tensor, param)
            print(f"Left parameters is {np.count_nonzero(masked_param.data.cpu().numpy())}, parameters has reduced {(np.count_nonzero(param.data.cpu().numpy()) - np.count_nonzero(masked_param.data.cpu().numpy())) / (np.count_nonzero(param.data.cpu().numpy()) + 1e-3) * 100} %.")
            masked_param_list.append(masked_param)
        self.learn.opt.param_groups[0]['params'].items = masked_param_list

    def before_batch(self):
        if self.every_batch:
            self.before_fit()
        else:
            pass


class Prune(Callback):
    def __init__(self, alpha=0.1, k_step=5, min_sparsity=0.1, min_param_count=100):
        """
        在参数的稀疏数量 > min_sparsity 且绝对值 > min_param_count 的情况下进行剪枝
        """
        super(Prune, self).__init__()
        self.alpha = alpha
        self.k_step = k_step
        self.total_step = 0
        self.min_sparsity = min_sparsity
        self.min_param_count = min_param_count
        self.stop = False

    def before_train(self):
        print("Computing masks...")
        self.param_list = self.learn.opt.param_groups[0]['params'].items
        self.masks = [np.ones_like(x.data.cpu()) for x in self.param_list]
        self.param_count = self.count_param(self.param_list)

    def prune_by_percent_once(self, percent, mask, final_weight):
        final_weight_ = final_weight.data.cpu().numpy()
        sorted_weights = np.sort(np.abs(final_weight_[mask == 1]))
        weights_count = self.count_param(final_weight)
        mask_count = np.count_nonzero(mask)
        sparsity = mask_count / weights_count
        if sparsity > self.min_sparsity and mask_count > self.min_param_count:
            cutoff_index = np.round(percent * sorted_weights.size).astype(int)
            cutoff = sorted_weights[cutoff_index]
        else:
            cutoff = -np.inf  # 若不满足条件，则将阈值调整至-inf，也就是不做剪枝
        return np.where(np.abs(final_weight_) <= cutoff, np.zeros(mask.shape), mask)

    def count_param(self, param):
        if isinstance(param, list):
            count = 0
            for i in param:
                tmp = 1
                for j in i.shape:
                    tmp *= j
                count += tmp
        else:
            count = 1
            for i in param.shape:
                count *= i
        return count

    def before_batch(self):
        self.param_list = self.learn.opt.param_groups[0]['params'].items
        masked_param_list = []
        for ind, param in enumerate(self.param_list):
            # print(f"Layer {ind}: ")
            # print(f"Original param is {np.count_nonzero(param.data.cpu().numpy())}")
            mask = self.masks[ind]
            mask_tensor = torch.tensor(mask)
            param = param.cpu()
            masked_param = torch.mul(mask_tensor, param)
            masked_param = masked_param.cuda()
            # print(f"Masked parameters is {np.count_nonzero(masked_param.data.cpu().numpy())}, parameters has reduced {(np.count_nonzero(param.data.cpu().numpy()) - np.count_nonzero(masked_param.data.cpu().numpy())) / (np.count_nonzero(param.data.cpu().numpy()) + 1e-3)} %.")
            masked_param_list.append(masked_param)
        self.learn.opt.param_groups[0]['params'].items = masked_param_list

    def after_batch(self):
        self.total_step += 1
        if self.total_step % self.k_step == 0 and not self.stop:
            print("Starting cutting param...")
            for ind, param in enumerate(self.param_list):
                self.masks[ind] = self.prune_by_percent_once(percent=self.alpha, mask=self.masks[ind], final_weight=param)
            mask_count = 0
            for mask in self.masks:
                mask_count += np.count_nonzero(mask)
            if (self.param_count - mask_count) / self.param_count > 1 - self.min_sparsity:
                self.stop = True
            print(f"Original parameter num: {self.param_count}, final parameter num: {mask_count}, parameter has reduced {(self.param_count - mask_count) / self.param_count * 100} %.")
        elif self.stop:
            torch.save(self.masks, os.path.join(root, 'ckpt', f'mask_{self.learn.model.name}_{self.total_step}.pth'))
            raise CancelFitException()


class CountParams(Callback):
    def count_param(self, param):
        count = 0
        for i in param:
            i = i.data.cpu().numpy()
            tmp = 1
            for j in i.shape:
                tmp *= j
            count += tmp
        return count

    def count_nonzero_param(self, param):
        count = 0
        for i in param:
            i = i.data.cpu().numpy()
            tmp = np.count_nonzero(i)
            count += tmp
        return count

    def before_batch(self):
        param_list = self.learn.opt.param_groups[0]['params'].items
        all_param_num = self.count_param(param_list)
        non_zero_param_num = self.count_nonzero_param(param_list)
        sparsity = (all_param_num - non_zero_param_num) / all_param_num
        print(f"All param: {all_param_num}, non-zero param: {non_zero_param_num}, sparsity: {sparsity}.")