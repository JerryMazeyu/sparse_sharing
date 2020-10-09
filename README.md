# sparse_sharing
基于fastai实现稀疏共享与改进的彩票假设

### 基于fastai实现的[彩票假设](https://arxiv.org/pdf/1803.03635.pdf)

可以在xresnet18上修剪90%以上的参数下保持和原始参数同样的accuracy

通过在`train.py`中选择不同的callbacks实现

1. 初始训练

   选择cbs为第一个，运行`python train.py -n <your_learner_index> -e <epoches>`

2. 修剪网络

   选择cbs为第二个，运行`python train.py -n <same as step 1> -e 1`

3. 测试准确率

   选择cbs为第三个，运行`python train.py -n <same as step 1> -e 1 -m eval`

