import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader

# 自己的工具包
from nnNetwork import CNN  # 神经网络模型
from trainTools import train_ch5  # 训练模型的
from ProcessData import getDataFromDir  # 制作数据的
from ProcessData import EcgDataset

#经典网络模型
from model_cfg.densenet import densenet121,densenet161,densenet169,densenet201
from model_cfg.mobilenetv2 import mobilenet_v2
from model_cfg.mobilenetv3 import mobilenet_v3_small,mobilenet_v3_large
# from model_cfg.vision_transformer import vit_b_16,vit_l_16,vit_b_32,vit_l_32
from model_cfg.resnet import resnet18,resnet34,resnet50,resnet101,resnet152

"""为了统计数据"""
from createTable import crTable  # 统计数据的

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PS 模型的设定参数要在nnNetwork里面修改
# 模型训练参数设定
wd = 0.0001
lr = 0.00123
num_epochs = 30

# 路径
PATH = "./net.pt"  # PATH是模型保存路径

# 数据文件地址是参数输入
train_data_path = 'data/origin/origin_baseline_train'
test_data_path = 'data/origin/origin_filter_test'
if len(sys.argv) > 2 and sys.argv[2] == 'ob':
    train_data_path = 'data/origin/origin_baseline_train'
elif len(sys.argv) > 2 and sys.argv[2] == 'of':
    train_data_path = 'data/origin/origin_filter_train'
    test_data_path = 'data/origin/origin_filter_test'
elif len(sys.argv) > 2 and sys.argv[2] == 'hpb':
    train_data_path = 'data/origin/half_patient_baseline'
elif len(sys.argv) > 2 and sys.argv[2] == 'hpf':
    train_data_path = 'data/origin/half_patient_filter'
elif len(sys.argv) > 2 and sys.argv[2] == '8k':
    train_data_path = 'data/mix8000b'
elif len(sys.argv) > 2 and sys.argv[2] == '15k':
    train_data_path = 'data/mix15000b'
elif len(sys.argv) > 2 and sys.argv[2] == '8kpf':
    train_data_path = 'data/mix8000pf'

# Flatten layer展平（之后尝试去掉，自己手动前向）

model_save_name = 'e_model'
model_save_iteration = -1
csv_save_name = 'e_model'

net = CNN()
train_data = EcgDataset(train_data_path, "resnet")
test_data = EcgDataset(test_data_path, "resnet")

if len(sys.argv) > 1 and sys.argv[1] == 'd121':
    train_data = EcgDataset(train_data_path, "densenet")
    test_data = EcgDataset(test_data_path, "densenet")
    # desenet
    net = densenet121(pretrained=False)
    model_save_name = 'd121'
    csv_save_name = 'd121'
    # net = densenet161(pretrained=False)
    # net = densenet169(pretrained=False)
    # net = densenet201(pretrained=False)
elif len(sys.argv) > 1 and sys.argv[1] == 'rs18':
    train_data = EcgDataset(train_data_path, "resnet")
    test_data = EcgDataset(test_data_path, "resnet")
    # resent
    net = resnet18(pretrained=False)
    model_save_name = 'rs18'
    csv_save_name = 'rs18'
    # net = resnet50(pretrained=False)
    # net = resnet101(pretrained=False)
    # net = resnet152(pretrained=False)
elif len(sys.argv) > 1 and sys.argv[1] == 'rs34':
    train_data = EcgDataset(train_data_path, "resnet")
    test_data = EcgDataset(test_data_path, "resnet")
    net = resnet34(pretrained=False)
    model_save_name = 'rs34'
    csv_save_name = 'rs34'
elif len(sys.argv) > 1 and sys.argv[1] == 'mbv2':
    train_data = EcgDataset(train_data_path, "mobilenet")
    test_data = EcgDataset(test_data_path, "mobilenet")
    # mobilenet
    net = mobilenet_v2(pretrained=False)
    model_save_name = 'mbv2'
    csv_save_name = 'mbv2'
    # net = mobilenet_v3_small(pretrained=False)
    # net = mobilenet_v3_large(pretrained=False)

# vision transformer
# net = vit_b_16(pretrained=False)
# net = vit_b_32(pretrained=False)
# net = vit_l_16(pretrained=False)
# net = vit_l_32(pretrained=False)

# num_workers是额外进程数目
num_workers = 2
batch_size = 32
train_iter = DataLoader(train_data, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers,
                        drop_last=False)
test_iter = DataLoader(test_data, batch_size=batch_size,
                       shuffle=True, num_workers=num_workers)


# optimizer = torch.optim.SGD(net.parameters(), lr=0.00123,weight_decay=0) #学习速率调整到0.01很重要
loss = nn.CrossEntropyLoss()

train_ch5(net, train_iter, test_iter, batch_size, lr, wd, device, num_epochs, csv_save_name)

# save model
torch.save(net, model_save_name + '.pt')


"""统计数据
"""
# crTable(net, X_test, Y_test, csvSaveName, device)修改为
# crTable(net, test_iter, csvSaveName, device)
# print(csvSaveName)
# crTable(net,test_iter,csvSaveName,device)
# print(csvSaveName)
