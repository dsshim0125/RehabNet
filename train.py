import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from RehabNet import GRU
import argparse
from data_utils import data_loader
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', default=100, type=int)
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--hidden_size', default=10, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--input_size', default=1, type=int)
parser.add_argument('--num_layers', default=10, type=int)
parser.add_argument('--is_training', default=False, type=str2bool)

args = parser.parse_args()
index = args.index
window_size = args.window_size
hidden_size = args.hidden_size
learning_rate = args.lr
batch_size = args.batch_size
input_size = args.input_size
num_layers = args.num_layers
is_training = args.is_training

data = data_loader(window_size=window_size, index=index)


if is_training:

    train_time, train_data, train_label = data.train_data_loader()
    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.float32)
    train_dataset = TensorDataset(train_data, train_label)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=1, device=device)

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 100
    total_step = len(train_loader)
    model.to(device)
    for epoch in range(num_epochs):
        total_loss=0
        for input, labels in tqdm.tqdm(train_loader):
            input = input.view(-1, window_size, input_size).to(device)
            labels = labels.to(device).view(-1,1)


            outputs = model(input)

            loss = loss_fn(outputs, labels)
            total_loss += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch:%d, Avg.loss:%.5f'%(epoch+1, total_loss/total_step))

        torch.save(model.state_dict(), 'model.ckpt')
    print('Training Done!')

else:

    device = torch.device('cuda')
    model = GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=1, device=device)
    model = model.to(device)
    model.load_state_dict(torch.load('model.ckpt'))
    model.eval()

    train_time, train_label, train_data = data.plotting_train_data()
    test_time, test_label, test_data = data.plotting_test_data()

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_label = torch.tensor(train_label, dtype=torch.float32)

    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_label = torch.tensor(test_label, dtype=torch.float32)

    train_dataset = TensorDataset(train_data, train_label)
    test_dataset = TensorDataset(test_data, test_label)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    train_prediction = np.empty([1, 1])
    test_prediction = np.empty([1, 1])

    time_start = time.time()

    for input, label in train_loader:

        train_data = input.view(-1, window_size, 1).to(device)
        train_predict = model(train_data)

        train_predict = train_predict.detach().to('cpu').numpy()

        train_prediction = np.vstack((train_prediction, train_predict))


    train_prediction = train_prediction[1:]
    train_label = train_label.numpy()

    train_inference_time = (time.time() - time_start)/(train_time[-1]-train_time[0])

    for input, label in test_loader:
        test_data = input.view(-1, window_size, 1).to(device)
        test_predict = model(test_data)

        test_predict = test_predict.detach().to('cpu').numpy()

        test_prediction = np.vstack((test_prediction, test_predict))

    test_inference_time = time.time() - time_start
    test_prediction = test_prediction[1:]
    test_label = test_label.numpy()

    print('Inference time = %.3f/sec' % train_inference_time)

    plt.figure(figsize=(15,8))
    plt.subplot(2,1,1)
    plt.plot(train_time, train_label)
    plt.scatter(train_time, train_prediction, c='r')
    plt.title('Insample Inference')

    plt.subplot(2,1,2)
    plt.plot(test_time, test_label)
    plt.scatter(test_time, test_prediction, c='r')
    plt.title('Outsample Inference')

    plt.show()


