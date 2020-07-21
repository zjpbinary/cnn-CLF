import torch
import torch.nn as nn

from loaddata import *
from textCNN import *
from config import *

def cnn_train(trainpairs, model, criterion, optimizer, device, epoch):
    model.train()
    for i in range(epoch):
        for j, pair in enumerate(trainpairs):
            optimizer.zero_grad()
            target = torch.LongTensor(pair[0]).to(device)
            input = torch.LongTensor(pair[1]).to(device)
            output = model.forward(input)

            loss =criterion(output, target)
            loss.backward()
            optimizer.step()
            print('第%d次迭代'%i, '第%d个batch'%j, 'loss为：', loss)
def cnn_predict(testdata, model, device):
    model.eval()
    right = 0.
    for i in range(len(testdata[0])):
        predict = model.forward(torch.LongTensor(testdata[1][i]).unsqueeze(0).to(device))
        predict = torch.argmax(predict)
        if predict.item()== testdata[0][i]:
            right+=1
    precision = right/len(testdata[0])
    print('测试集上的精度为：', precision)
if __name__ == '__main__':
    args = TextCNNConfig()
    vocab_size, label_size, trainpairs, testdata = load_data(args.trainfile, args.testfile, args.batch_size)
    #print(vocab_size)
    #device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device)
    cnn = TextCNN(vocab_size, label_size, args.d_model, args.dropout, filter_num=args.filter_num, filter_sizes=args.filter_sizes).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    lr = args.lr
    optimizer = torch.optim.Adam(cnn.parameters(), lr)
    cnn_train(trainpairs, cnn, criterion, optimizer, device, args.epoch)
    cnn_predict(testdata, cnn, device)



