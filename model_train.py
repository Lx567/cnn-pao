from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import torch
import torch.nn as nn
from model_VGG16 import VGG16
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
import pandas as pd


def train_val_data_process():
    data_set = FashionMNIST(root='./data',
                            train=True,
                            transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]),
                            download=True)
    train_data, val_data = Data.random_split(data_set,[round(len(data_set)*0.8),round(len(data_set)*0.2)])
    train_dataloader = Data.DataLoader(train_data,batch_size=64,shuffle=True,num_workers=4)
    val_dataloader = Data.DataLoader(val_data,batch_size=64,shuffle=True,num_workers=4)

    return train_dataloader,val_dataloader

def train_model_process(model,train_dataloader,val_dataloader,num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    # 初始化参数
    best_acc = 0.0
    train_loss_all = []
    val_loss_all = []
    train_acc_all = []
    val_acc_all = []

    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # 初始化参数
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0
        val_corrects = 0
        train_num = 0
        val_num = 0

        for step,(b_x,b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.train()
            output = model(b_x)
            pre_lab = torch.argmax(output,dim=1)
            loss = criterion(output,b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 更新参数
            train_loss += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab==b_y.data)
            train_num += b_x.size(0)
        for step,(b_x,b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output,dim=1)
            loss = criterion(output,b_y)
            val_loss += loss.item() * b_x.size(0)
            val_corrects += torch.sum(pre_lab==b_y.data)
            val_num += b_x.size(0)
        
        train_loss_all.append(train_loss/train_num)
        val_loss_all.append(val_loss/val_num)
        train_acc_all.append(train_corrects.double()/train_num)
        val_acc_all.append(val_corrects.double()/val_num)
        print("{} Train Loss: {:.4f} Train Acc: {:.4f}".format(epoch,train_loss_all[-1],train_acc_all[-1]))
        print("{} Val Loss: {:.4f} Val Acc: {:.4f}".format(epoch,val_loss_all[-1],val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            torch.save(model.state_dict(),"VGG16.pth")
            print("Saved model!")        
        time_used = time.time() - since
        print("Time used: {:.0f}m {:.0f}s".format(time_used//60,time_used%60))

    train_process = pd.DataFrame(data={'epoch':range(num_epochs),
                                        'train_loss_all':train_loss_all,
                                        'val_loss_all':val_loss_all,
                                        'train_acc_all':train_acc_all,
                                        'val_acc_all':val_acc_all})
    return train_process

def  matplot_acc_loss(train_process):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process['epoch'],train_process['train_loss_all'],'ro-',label='train_loss')
    plt.plot(train_process['epoch'],train_process['val_loss_all'],'bs-',label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(1,2,2)
    plt.plot(train_process['epoch'],train_process['train_acc_all'],'ro-',label='train_acc')
    plt.plot(train_process['epoch'],train_process['val_acc_all'],'bs-',label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('accuracy')

    plt.savefig('./train_vgg16.png')
    plt.show()


if __name__ == '__main__':
    model = VGG16()
    train_dataloader,val_dataloader = train_val_data_process()
    train_process = train_model_process(model,train_dataloader,val_dataloader,num_epochs=20)
    matplot_acc_loss(train_process)