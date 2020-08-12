from RehabNet import GRU, Discriminator
from data_utils import data_loader
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--window_size', default=100, type=int)
parser.add_argument('--index', default=0, type=int)
parser.add_argument('--hidden_size', default=10, type=int)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--input_size', default=1, type=int)
parser.add_argument('--num_layers', default=10, type=int)

args = parser.parse_args()
index = args.index
window_size = args.window_size
hidden_size = args.hidden_size
learning_rate = args.lr
batch_size = args.batch_size
input_size = args.input_size
num_layers = args.num_layers

data = data_loader(window_size=window_size, index=index)

train_time, train_data, train_label = data.train_data_loader()
train_data = torch.tensor(train_data, dtype=torch.float32)
train_label = torch.tensor(train_label, dtype=torch.float32)
train_dataset = TensorDataset(train_data, train_label)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=1, device=device)
generator = generator.to(device)
generator.load_state_dict(torch.load('model.ckpt'))

discriminator = Discriminator()
discriminator = discriminator.to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

loss_fn = nn.BCEWithLogitsLoss()
ones = torch.ones(batch_size).to(device)
zeros = torch.zeros(batch_size).to(device)


num_epochs = 100
for epoch in range(num_epochs):
    for input, labels in tqdm.tqdm(train_loader):
        input = input.view(-1, window_size, 1).to(device)
        labels = labels.to(device).view(-1, 1).to(device)

        outputs = generator(input)
        outputs_copy = outputs.detach()

        model_scores = discriminator(outputs)
        model_scores = model_scores.view(-1)
        loss_G = loss_fn(model_scores, ones[:input.size(0)])

        discriminator.zero_grad(), generator.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        label_scores = discriminator(labels)
        label_scores = label_scores.view(-1)
        loss_D_label = loss_fn(label_scores, ones[:input.size(0)])

        model_scores = discriminator(outputs_copy)
        model_scores = model_scores.view(-1)
        loss_D_model = loss_fn(model_scores, zeros[:input.size(0)])

        loss_D = loss_D_label + loss_D_model

        discriminator.zero_grad(), generator.zero_grad()
        loss_D.backward()
        optimizer_D.step()

    print ('Epoch:%d, Loss_D:.%3f, Loss_G:.%3f'%(epoch+1, loss_D, loss_G))
    torch.save(generator.state_dict(), 'model_adv.ckpt')
print('Training_Done')
