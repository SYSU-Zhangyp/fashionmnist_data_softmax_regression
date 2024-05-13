import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import load_data as ld
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 从数据集加载数据
train_feature, train_label, test_feature, test_label = ld.load_mnist()

# 超参数
input_size = 28 * 28  # 输入特征数量
num_classes = 10      # 类别数量
learning_rate = 0.1  # 学习率
batch_size = 256      # mini-batch 大小
num_epochs = 80       # 迭代次数
num_workers = 0       # 多线程读取数据

# 定义 softmax 回归模型
class SoftmaxRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.input_size = input_size
        self.linear = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x = x.view(-1, input_size)  # 将输入展平
        return self.linear(x)

def train(model, train_loader, test_loader):  
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss  
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)  # SGD optimizer  
    best_accuracy = 0.0  # Initialize best accuracy
    best_model_wts = copy.deepcopy(model.state_dict())  # Initial copy of best model weights
  
    # Training the model
    model.train()
    total_step = len(train_loader)  
    for epoch in range(num_epochs):  
        for i, (images, labels) in enumerate(train_loader):  
            # Forward pass  
            outputs = model(images)  
            loss = criterion(outputs, labels)  
              
            # Backward and optimize  
            optimizer.zero_grad()  
            loss.backward()  
            optimizer.step()  
              
            # Track accuracy
            if (i+1) % 50 == 0:
                accuracy = test(model, test_loader)
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item(), accuracy))
                # Check if current model has highest accuracy
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_wts = copy.deepcopy(model.state_dict())  # Update best model weights
    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Save the best model parameters
    torch.save(model.state_dict(), 'softmax_regression_best_model.ckpt')  
    print('Training completed, Best Accuracy: {:.2f}%'.format(best_accuracy))

def test(model, test_loader):
    model.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('测试集准确率: {} %'.format(accuracy))
    return accuracy

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Normalize each row
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.set_theme(font_scale=1.2)
    sns.heatmap(cm, annot=True, annot_kws={"size": 12}, fmt='.2f', cmap=cmap, cbar=False, xticklabels=classes, yticklabels=classes)
    
    plt.title(title)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Print probabilities for each row and column
    for i in range(len(classes)):
        if i == 0:
            continue
        for j in range(len(classes)):
            if normalize:
                text = '{:.2f}'.format(cm[i, j])
            else:
                text = '{:d}'.format(cm[i, j])
            plt.text(j + 0.5, i + 0.5, text, horizontalalignment='center', verticalalignment='center', fontsize=12)

    plt.show()

def show_plot(model):
    # 在测试集上进行预测
    model.eval()
    predicted_labels = []
    true_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    # 定义类别
    classes = [str(i) for i in range(num_classes)]

    print(classes)

    # 绘制混淆矩阵
    plot_confusion_matrix(true_labels, predicted_labels, classes, normalize=True)

if __name__=="__main__":
    # 定义数据加载器
    train_dataset = Data.TensorDataset(train_feature, train_label)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

    # 定义测试集的数据加载器
    test_dataset = Data.TensorDataset(test_feature, test_label)
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    # 创建模型
    model = SoftmaxRegression(input_size, num_classes)

    train(model, train_loader, test_loader)
    test(model, test_loader)
    show_plot(model)


