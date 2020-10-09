# append root
import os
import sys
a = (os.path.abspath(sys.argv[0])[0: os.path.abspath(sys.argv[0]).find('sparse_sharing')])
b = 'sparse_sharing'
root = os.path.join(a, b)
sys.path.append(root)
# import fastai and torch
from fastai.vision.all import accuracy, Learner
import torch
# import model
from models.models import *
# import callback
from src.callbacks import *
# import dataloader
from src.build_dataloader import dataloader_s224_bs16, dataloader_s512_bs4, dataloader_s312_bs12
# import misc
import argparse

# original train
# cbs = [ChooseTask, SaveModelCallback1(monitor='accuracy'), ReduceLROnPlateau(factor=3)]
# iterative prune
# cbs = [ChooseTask, LoadCkpt(ckpt_name='tst_bs16_origin', every_batch=True), Prune]
# test
cbs = [ChooseTask, LoadCkpt(ckpt_name='tst_bs16_origin'), LoadMask(mask_name='mask_tst.pth'), CountParams]

# train
if torch.cuda.is_available():
    learner1 = Learner(dataloader_s512_bs4.cuda(), xse_resnet50.cuda(), metrics=accuracy, cbs=cbs)
    learner2 = Learner(dataloader_s512_bs4.cuda(), efficient_b6.cuda(), metrics=accuracy, cbs=cbs)
    learner3 = Learner(dataloader_s224_bs16.cuda(), xse_resnet50.cuda(), metrics=accuracy, cbs=cbs)
    learner4 = Learner(dataloader_s224_bs16.cuda(), efficient_b6.cuda(), metrics=accuracy, cbs=cbs)
    learner5 = Learner(dataloader_s312_bs12.cuda(), xse_resnet50.cuda(), metrics=accuracy, cbs=cbs)
    learner6 = Learner(dataloader_s312_bs12.cuda(), efficient_b6.cuda(), metrics=accuracy, cbs=cbs)
    learner7 = Learner(dataloader_s224_bs16.cuda(), tst_model.cuda(), metrics=accuracy, cbs=cbs)
else:
    learner1 = Learner(dataloader_s512_bs4, xse_resnet50, metrics=accuracy, cbs=cbs)
    learner2 = Learner(dataloader_s512_bs4, efficient_b6, metrics=accuracy, cbs=cbs)
    learner3 = Learner(dataloader_s224_bs16, xse_resnet50, metrics=accuracy, cbs=cbs)
    learner4 = Learner(dataloader_s224_bs16, efficient_b6, metrics=accuracy, cbs=cbs)
    learner5 = Learner(dataloader_s312_bs12, xse_resnet50, metrics=accuracy, cbs=cbs)
    learner6 = Learner(dataloader_s312_bs12, efficient_b6, metrics=accuracy, cbs=cbs)
    learner7 = Learner(dataloader_s224_bs16, tst_model, metrics=accuracy, cbs=cbs)

learner_list = [learner1, learner2, learner3, learner4, learner5, learner6, learner7]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Im2Latex Training Program")
    parser.add_argument("-n", dest="num", default=7, type=int, help="选择的learner")
    parser.add_argument("-e", dest="epoch", default=10, type=int)
    parser.add_argument("-m", dest="mode", default="train", choices=["train", "eval"])
    args = parser.parse_args()
    target_learner = learner_list[args.num - 1]
    if args.mode == "eval":
        target_learner.train_eval.run_train = False
        target_learner.train_eval.run_valid = True
    target_learner.fit_one_cycle(args.epoch, 1e-2)
