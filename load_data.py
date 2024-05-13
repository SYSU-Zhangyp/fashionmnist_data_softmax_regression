import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的⽂本标签。 """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_mnist():
    # 加载数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root='~/fashionmnist_data/FashionMNIST/train',
        train=True,
        download=True,
        transform=transforms.ToTensor())

    mnist_test = torchvision.datasets.FashionMNIST(
        root='~/fashionmnist_data/FashionMNIST/test',
        train=False,
        download=True,
        transform=transforms.ToTensor())    
    
    #访问样本
    train_feature = torch.stack([img for img, _ in mnist_train], dim=0)
    train_label = torch.tensor([label for _, label in mnist_train]) 

    test_feature = torch.stack([img for img, _ in mnist_test], dim=0)
    test_label = torch.tensor([label for _, label in mnist_test]) 

    # train_feature = train_feature.view(-1, 784)
    # test_feature = test_feature.view(-1,784)
    return train_feature, train_label, test_feature, test_label

if __name__ == "__main__":
    # 加载数据集
    mnist_train = torchvision.datasets.FashionMNIST(
        root='~/fashionmnist_data/FashionMNIST/train',
        train=True,
        download=True,
        transform=transforms.ToTensor())

    mnist_test = torchvision.datasets.FashionMNIST(
        root='~/fashionmnist_data/FashionMNIST/test',
        train=False,
        download=True,
        transform=transforms.ToTensor())

    print(type(mnist_train))
    print(len(mnist_train), len(mnist_test))

    #访问样本
    feature, label = mnist_train[0]
    print(feature.shape, label) # Channel x Height X Width

    num_inputs = 784
    X = feature.view(-1, num_inputs)
    print(X.shape)

    # 展示图像及其标签
    X, y = [], []
    for i in range(10):
        X.append(mnist_train[i][0])
        y.append(mnist_train[i][1])
    show_fashion_mnist(X, get_fashion_mnist_labels(y))
    
