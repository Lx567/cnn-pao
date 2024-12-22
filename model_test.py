from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import torch
from model_VGG16 import VGG16
import matplotlib
matplotlib.use('agg')


def test_data_process():
    data_set = FashionMNIST(root='./data',
                            train=False,
                            transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()]),
                            download=True)
    test_dataloader = Data.DataLoader(data_set,batch_size=64,shuffle=True,num_workers=4)

    return test_dataloader

def train_model_process(model,test_dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 初始化参数
    test_correct = 0
    test_num = 0
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    with torch.no_grad():
        for b_x,b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            model.eval()
            output = model(b_x)
            pre_lab = torch.argmax(output,dim=1)
            result = pre_lab[0]
            label = b_y[0]
            print("Predict: {} Label: {}   {}".format(classes[result], classes[label], "√" if result == label else "x"))
            test_correct += torch.sum(pre_lab==b_y.data)
            test_num += b_x.size(0)
        test_acc = test_correct.double().item()/test_num
        print("Test Acc: {:.4f}".format(test_acc))

if __name__ == '__main__':
    model = VGG16()
    model.load_state_dict(torch.load('VGG.pth'))
    test_dataloader = test_data_process()
    train_model_process(model,test_dataloader)
