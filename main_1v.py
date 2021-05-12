#在python3中训练
import argparse
import os
import time
import pickle
import sys

print(sys.path)
#sys.path.append('/home/wgk/code/PointNetGPD/model')


import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
#from tensorboardX import SummaryWriter
#现在pytorch已经集成了tensorboard了
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

##################/home/wgk/code/PointNetGPD/PointNetGPD/model 包###########################

#需要安装pcl
from model.dataset import *
#pointnet的具体网络结构
from model.pointnet import PointNetCls, DualPointNetCls


import argparse

#创建一个argparse模块中的ArgumentParser类对象
# description只是解释一下，这个类对象是做什么的
parser = argparse.ArgumentParser(description='pointnetGPD')
#添加一个参数  tag  这个可以自定义，主要用于区分指定训练结果的存放文件夹等
parser.add_argument('--tag', type=str, default='default')
#默认epoch训练的轮数，大小为200
parser.add_argument('--epoch', type=int, default=200)
#添加模式参数，指定从train或者test中选择，且是必选项
parser.add_argument('--mode', choices=['train', 'test'], required=True)
#设定batch-size，默认是1
parser.add_argument('--batch-size', type=int, default=1)
#意思是，当命令中出现 --cuda的时候，就把  object.cuda的值设置为真
parser.add_argument('--cuda', action='store_true')
#添加参数  选择使用的gpu编号，默认使用0号gpu
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--load-model', type=str, default='')


parser.add_argument('--load-epoch', type=int, default=-1)
parser.add_argument('--model-path', type=str, default='./assets/learned_models',
                   help='pre-trained model path')
parser.add_argument('--data-path', type=str, default=os.environ['HOME']+'/dataset/PointNetGPD', help='data path')
#设置每隔多少次迭代，就打印出来一次loss还有训练进度
parser.add_argument('--log-interval', type=int, default=10)
#保存间隔，设置每隔几个epoch就保存一次模型，比如epoch=10就是训练10轮之后就保存当前的训练模型
parser.add_argument('--save-interval', type=int, default=1)


#此时，使用对象parser中的解析函数.parse_args()，读取命令行给出的参数，集合成namspace  返回给args
#例子：    python main_1v.py --epoch 200 --mode train --batch-size 3
args = parser.parse_args()

#查看是否安装好了cuda
args.cuda = args.cuda if torch.cuda.is_available else False
#为当前的gpu设置随机种子；这样在以后运行该程序的时候，随机数都是相同的，不会每次运行都变化一次
#作用主要是为了固定随机初始化的权重值，这样就可以在每次重新从头训练网络的时候，权重值虽然是随机的
#但是都是固定的，不会每次都在变化
#其中的seed值，可以随便写
"""
 https://blog.csdn.net/weixin_43002433/article/details/104706950?utm_
medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1
.add_param_isCf&depth_1-utm_source=distribute.pc_relevant.none-task-blog-Blog
CommendFromMachineLearnPai2-1.add_param_isCf
"""
if args.cuda:
    torch.cuda.manual_seed(1)

#获取当前文件所在目录（文件夹）的绝对路径
path=os.path.dirname(os.path.abspath(__file__))
#更改（确保）当前所在目录是工作目录
os.chdir(path)
print(os.getcwd())
#获取当前的时间，年月日  时间
current_time = time.strftime("%Y-%m-%dT%H:%M", time.localtime())
logger = SummaryWriter(os.path.join('./assets/log/', current_time))

#设置numpy的随机数种子，但是注意，这里的随机数种子是随着系统时间变化的，因此每次运行出现的随机数都是不同的
np.random.seed(int(time.time()))

"""
使用单线程进行数据集导入时候，有时候比较慢，会阻碍到计算的过程，于是考虑用多个线程进行数据的导入
如果参数num_workers设置的大于1，就是指定了多个线程，于是考虑使用多个线程导入数据，防止影响计算
每个线程叫一个"worker"，这个 worker_init_fn就是定义每个线程worker初始化的时候，需要执行哪些操作
其中，pid就是子线程的线程号，直接写成这样就行
"""
def worker_init_fn(pid):
    """
    为后台工作进程设置唯一但具有确定性的随机种子，只需要对numpy设置种子就行了，
    pytorch和python的随机数生成器会自己管理自己的子线程？
    设置torch.initial_seed() % (2**31-1)取余数，是因为numpy的种子范围是0到2**31-1
    """
    np.random.seed(torch.initial_seed() % (2**31-1))


"""
将样本采样器返回的list中对应的样构建成一个minibatch的tensor，自定义的tensor的话
对于map型的数据集，且有多个读取进程的情况，采样器会先将样本集分成一个个batch，返回每个batch的样本的索引的 list，
再根据my_collate函数，说明如何把这些索引指定的样本整合成一个data Tensor和lable Tensor
"""
def my_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

#
grasp_points_num=750
thresh_good=0.6
thresh_bad=0.6
#设置点是只有xyz  ？
point_channel=3


"""
数据加载器,主要实现数据加载到网络中的相关作用，核心类是torch.utils.data.DataLoader
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

dataset：加载的数据集，继承自torch.utils.data.Dataset类的自定义类的对象，或者就是torch.utils.data.Dataset类的对象
batch_size：batch size，就是batchs ize
shuffle:：是否将数据打乱
sampler： 样本抽样，后续会详细介绍
batch_sampler，从注释可以看出，其和batch_size、shuffle等参数是互斥的，一般采用默认
num_workers：使用多进程加载的进程数，0代表不使用多进程
collate_fn： 如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
pin_memory：是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
drop_last：dataset中的数据个数可能不是batch_size的整数倍，drop_last为True会将多出来不足一个batch的数据丢弃

"""
train_loader = torch.utils.data.DataLoader(
    #设置dataset，传入的是继承了torch.utils.data.Dataset类的子类的实例对象 class PointGraspOneViewDataset(torch.utils.data.Dataset):
    PointGraspOneViewDataset(
        #设置夹爪内部的点数量最少要有多少个
        grasp_points_num=grasp_points_num,
        path=args.data_path,
        prefix='panda',
        tag='train',
        grasp_amount_per_file=16800,    #每个物体已经生成的抓取点云个数，140×120  （单物体的生成140个不同抓取姿态×单物体共有120个不同视角点云）
        thresh_good=thresh_good,
        thresh_bad=thresh_bad,
    ),
    #设置batch_size
    batch_size=args.batch_size,
    #设置使用多少个子线程来导入数据，如果设置为0，那就是直接把数据导入到主线程中
    num_workers=32,
    #如果设置为True，就使得Tensor数据最开始存放于内存的锁页内存中，这样将内存Tensor转移到GPU显存就会快一些
    # 当计算机内存充足的时候，选择True，如果不充足，可能使用到虚拟内存的时候，就写为False
    pin_memory=True,
    #如果设置为True，会默认构建一个乱序采样器（为False时，构建顺序采样器）；每次epoch之后都会把数据集打乱
    shuffle=True,

    #
    worker_init_fn=worker_init_fn,
    #collate_fn函数，将sampler返回的数据list合并成为一个整体的tensor，作为一个mini-batch
    collate_fn=my_collate,
    #设置，如何处理训练到最后，数据集长度不足一个batch_size时的数据，True就抛弃，否则保留
    drop_last=True,
)
"""
测试时候用的数据集加载器；这里的test数据集是用来测试训练好的网络的准确率
"""
test_loader = torch.utils.data.DataLoader(
    PointGraspOneViewDataset(
        grasp_points_num=grasp_points_num,
        path=args.data_path,
        prefix='panda',
        #设置标签为test
        tag='test',
        grasp_amount_per_file=500,   #
        thresh_good=thresh_good,
        thresh_bad=thresh_bad,
        with_obj=True,
    ),
    batch_size=args.batch_size,
    num_workers=32,
    pin_memory=True,
    shuffle=True,
    worker_init_fn=worker_init_fn,
    collate_fn=my_collate,
)

is_resume = 0
if args.load_model and args.load_epoch != -1:
    is_resume = 1
#如果是测试模式
if is_resume or args.mode == 'test':
    #加载网络结构和参数，加载到命令行 指定的gpu中
    model = torch.load(args.load_model, map_location='cuda:{}'.format(args.gpu))
 
    model.device_ids = [args.gpu]
    print('load model {}'.format(args.load_model))
#如果是训练模式，就加载模型
else:
    model = PointNetCls(num_points=grasp_points_num, input_chann=point_channel, k=2)



#如果命令行出现了cuda字眼（args.cuda将会自动设置为True）
if args.cuda:
    if args.gpu != -1:
        #设置为指定编号的gpu，使用指定编号的gpu进行训练
        torch.cuda.set_device(args.gpu)
        #将网络模型的所有参数等都转移到gpu中（刚指定的那个）
        model = model.cuda()
    else:
        #如果args.gpu=-1
        device_id = [0]
        #在这里选择使用哪个gpu
        torch.cuda.set_device(device_id[0])
        #这句话，使得原有的model（网络），重构成为一个新的model，这个模型能够调用多个gpu同时运行
        model = nn.DataParallel(model,device_id).cuda()
#选择adam优化器
optimizer = optim.Adam(model.parameters(), lr=args.lr)
"""
from torch.optim.lr_scheduler import StepLR
StepLR类主要用来调整学习率，lr=learning rate，让学习率随着epoch变化
构造函数 输入数据optimizer选择优化器；选择step_size=30, gamma=0.5
成员函数StepLR.step()是用来
"""
scheduler = StepLR(optimizer, step_size=30, gamma=0.5)


#训练函数，仅仅是一个epoch，其中包含了很多batch
def train(model, loader, epoch):
    #每一次epoch之前，都更新一下学习率
    scheduler.step()
    #调用train函数训练；这个model是提前设置好的pointnet
    model.train()
    
    torch.set_grad_enabled(True)
    correct = 0
    dataset_size = 0
    """
    注意，如果使用了多线程导入数据集的情况下，在当调用enumerate时，将会在此时在后台创建多线程导入数据
    loader是一个可迭代对象，索引是batch_idx，对象是(data, target)
    这里实现的效果是：一个一个batch的训练网络，直到一个epoch训练完毕
    """
    for batch_idx, (data, target) in enumerate(loader):
        #print(len(data),"data len is")
        """
        在实际的实验过程中发现，当某一个batch(比如剩下来的某一个batch)中间只含有一个sample
        那么很有可能会报错，这里判断一下本次的batch中是不是只有一个样本，如果只有一个样本
        那就跳过这个batch
        """
        if len(data) <=1:
            continue

        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        #如果使用cuda的话
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        #这里进行前向传播，并输出结果          但是这个output是什么
        output, _ = model(data)
        #计算loss
        loss = F.nll_loss(output, target)
        #反向传播，更新权重
        loss.backward()
        #进行学习率的优化
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        #设置每隔多少次迭代（batch）打印一次loss，这里默认是10次迭代
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t{}'.format(
            #当前第几个epoch， 当前训练到该epoch中的第几个batch，数据集总共有多大
            epoch, batch_idx * args.batch_size, len(loader.dataset),
            #计算当前训练百分比
            100. * batch_idx * args.batch_size / len(loader.dataset), loss.item(), args.tag))

            """在tensorboard上面绘制折线图
            def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
            参数含义:
            tag (string): 显示什么名称
            scalar_value (float or string/blobname): 需要打印出来（同时会保存下来）的变量
            global_step (int): 就是坐标系横轴的下标显示什么变量
            walltime (float): Optional override default walltime (time.time())
              with seconds after epoch of event
            """
            logger.add_scalar('train_loss',   
                                        loss.cpu().item(), #这里的loss.cpu是不是利用到了cpu？                           做了什么计算？
                                        batch_idx + epoch * len(loader))#横坐标下标是迭代次数，训练了几次epoch
            """len(loader)和len(loader.dataset)区别
            len(loader.dataset) 返回的是数据集中的sample的数量
            len(loader)      返回的是 len(loader.dataset)/batch_size向上取整  就是一个epoch中包含有多少个batch（一个epoch迭代多少次）
            """
    return float(correct)/float(dataset_size)

#测试训练好的网络
def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    test_loss = 0
    correct = 0
    dataset_size = 0
    da = {}
    db = {}
    res = []
    for data, target, obj_name in loader:
        dataset_size += data.shape[0]
        data, target = data.float(), target.long().squeeze()
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output, _ = model(data) # N*C
        test_loss += F.nll_loss(output, target, size_average=False).cpu().item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).long().cpu().sum()
        for i, j, k in zip(obj_name, pred.data.cpu().numpy(), target.data.cpu().numpy()):
            res.append((i, j[0], k))

    test_loss /= len(loader.dataset)
    acc = float(correct)/float(dataset_size)
    return acc, test_loss


def main():
    #
    if args.mode == 'train':
        for epoch_i in range(is_resume*args.load_epoch, args.epoch):
            #训练一个epoch，把网络模型、数据集、当前是第几轮epoch的编号，都写进去
            acc_train = train(model, train_loader, epoch_i)
            #训练完毕i，精度等于多少
            print('Train done, acc={}'.format(acc_train))
            #使用测试数据集进行测试
            acc, loss = test(model, test_loader)
            #打印测试出的精度和loss
            print('Test done, acc={}, loss={}'.format(acc, loss))
            #然后把训练的结果测试的结果，存放在logger中，之后使用折线图打印出来
            logger.add_scalar('train_acc', acc_train, epoch_i)
            logger.add_scalar('test_acc', acc, epoch_i)
            logger.add_scalar('test_loss', loss, epoch_i)
            if epoch_i % args.save_interval == 0:
                path = os.path.join(args.model_path, current_time + '_{}.model'.format(epoch_i))
                #如果满足要求了，就保存下来，注意这里，需要选定_use_new_zipfile_serialization=False
                torch.save(model, path,_use_new_zipfile_serialization=False)
                print('Save model @ {}'.format(path))
    else:
        print('testing...')
        acc, loss = test(model, test_loader)
        print('Test done, acc={}, loss={}'.format(acc, loss))

if __name__ == "__main__":
    main()
