import numpy as np
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data as data


datafile = 'g_16.npz'
imsize = 24
lr = 0.002
epochs = 30
k = 32
l_i = 2
l_h = 150
l_q = 10
batch_size = 128
save_path = "s{0}_k{1}.pth".format(imsize, k)


class GridworldData(data.Dataset):
    def __init__(self,
                 file,
                 imsize,
                 train=True,
                 transform=None,
                 target_transform=None):
        assert file.endswith('.npz')  # Must be .npz format
        self.file = file
        self.imsize = imsize
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or test set

        self.images, self.S1, self.S2, self.labels =  \
                                self._process(file, self.train)

    def __getitem__(self, index):
        img = self.images[index]
        s1 = self.S1[index]
        s2 = self.S2[index]
        label = self.labels[index]
        # Apply transform if we have one
        if self.transform is not None:
            img = self.transform(img)
        else:  # Internal default transform: Just to Tensor
            img = torch.from_numpy(img)
        # Apply target transform if we have one
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, int(s1), int(s2), int(label)

    def __len__(self):
        return self.images.shape[0]

    def _process(self, file, train):
        """Data format: A list, [train data, test data]
        Each data sample: label, S1, S2, Images, in this order.
        """
        with np.load(file, mmap_mode='r') as f:
            if train:
                images = f['arr_0']
                S1 = f['arr_1']
                S2 = f['arr_2']
                labels = f['arr_3']
            else:
                images = f['arr_4']
                S1 = f['arr_5']
                S2 = f['arr_6']
                labels = f['arr_7']
        # Set proper datatypes
        images = images.astype(np.float32)
        S1 = S1.astype(int)  # (S1, S2) location are integers
        S2 = S2.astype(int)
        labels = labels.astype(int)  # Labels are integers
        # Print number of samples
        if train:
            print("Number of Train Samples: {0}".format(images.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(images.shape[0]))
        return images, S1, S2, labels


def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out


def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim == 0
        x = x.item()
    if isinstance(x, float): rep = "%g" % x
    else: rep = str(x)
    return " " * (l - len(rep)) + rep


def get_stats(loss, predictions, labels):
    cp = np.argmax(predictions.cpu().data.numpy(), 1)
    error = np.mean(cp != labels.cpu().data.numpy())
    return loss.item(), error


def print_stats(epoch, avg_loss, avg_error, num_batches, time_duration):
    print(
        fmt_row(10, [
            epoch + 1, avg_loss / num_batches, avg_error / num_batches,
            time_duration
        ]))


def print_header():
    print(fmt_row(10, ["Epoch", "Train Loss", "Train Error", "Epoch Time"]))


class VIN(nn.Module):
    def __init__(self):
        super(VIN, self).__init__()
        self.h = nn.Conv2d(
            in_channels=l_i,
            out_channels=l_h,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=True)
        self.r = nn.Conv2d(
            in_channels=l_h,
            out_channels=1,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            bias=False)
        self.q = nn.Conv2d(
            in_channels=1,
            out_channels=l_q,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            bias=False)
        self.fc = nn.Linear(in_features=l_q, out_features=8, bias=False)
        self.w = Parameter(
            torch.zeros(l_q, 1, 3, 3), requires_grad=True)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input_view, state_x, state_y, k):
        """
        :param input_view: (batch_sz, imsize, imsize)
        :param state_x: (batch_sz,), 0 <= state_x < imsize
        :param state_y: (batch_sz,), 0 <= state_y < imsize
        :param k: number of iterations
        :return: logits and softmaxed logits
        """
        h = self.h(input_view)  # Intermediate output
        r = self.r(h)           # Reward
        q = self.q(r)           # Initial Q value from reward
        v, _ = torch.max(q, dim=1, keepdim=True)

        def eval_q(r, v):
            return F.conv2d(
                # Stack reward with most recent value
                torch.cat([r, v], 1),
                # Convolve r->q weights to r, and v->q weights for v. These represent transition probabilities
                torch.cat([self.q.weight, self.w], 1),
                stride=1,
                padding=1)

        # Update q and v values
        for i in range(k - 1):
            q = eval_q(r, v)
            v, _ = torch.max(q, dim=1, keepdim=True)

        q = eval_q(r, v)
        # q: (batch_sz, l_q, map_size, map_size)
        batch_sz, l_q, _, _ = q.size()
        q_out = q[torch.arange(batch_sz), :, state_x.long(), state_y.long()].view(batch_sz, l_q)

        logits = self.fc(q_out)  # q_out to actions

        return logits, self.sm(logits)

def train(net, trainloader, criterion, optimizer):
    print_header()
    # Automatically select device to make the code device agnostic
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # print(device)
    for epoch in range(epochs):  # Loop over dataset multiple times
        avg_error, avg_loss, num_batches = 0.0, 0.0, 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader):  # Loop over batches of data
            # Get input batch
            X, S1, S2, labels = [d.to(device) for d in data]
            if X.size()[0] != batch_size:
                continue  # Drop those data, if not enough for a batch
            net = net.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs, predictions = net(X, S1, S2, k)
            # Loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Update params
            optimizer.step()
            # Calculate Loss and Error
            loss_batch, error_batch = get_stats(loss, predictions, labels)
            avg_loss += loss_batch
            avg_error += error_batch
            num_batches += 1
        time_duration = time.time() - start_time
        # Print epoch logs
        print_stats(epoch, avg_loss, avg_error, num_batches, time_duration)
    print('\nFinished training. \n')


def test(net, testloader):
    total, correct = 0.0, 0.0
    # Automatically select device, device agnostic
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i, data in enumerate(testloader):
        # Get inputs
        X, S1, S2, labels = [d.to(device) for d in data]
        if X.size()[0] != batch_size:
            continue  # Drop those data, if not enough for a batch
        net = net.to(device)
        # Forward pass
        outputs, predictions = net(X, S1, S2, k)
        # Select actions with max scores(logits)
        _, predicted = torch.max(outputs, dim=1, keepdim=True)
        # Unwrap autograd.Variable to Tensor
        predicted = predicted.data
        # Compute test accuracy
        correct += (torch.eq(torch.squeeze(predicted), labels)).sum()
        total += labels.size()[0]
    print('Test Accuracy: {:.2f}%'.format(100 * (correct / total)))


if __name__ == '__main__':
    # Get path to save trained model
    
    # Instantiate a VIN model
    net = VIN()
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=lr, eps=1e-6)
    # Dataset transformer: torchvision.transforms
    transform = None
    # Define Dataset
    trainset = GridworldData(
        datafile, imsize=imsize, train=True, transform=transform)
    testset = GridworldData(
        datafile,
        imsize=imsize,
        train=False,
        transform=transform)
    # Create Dataloader
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)
    # Train the model
    train(net, trainloader, criterion, optimizer)
    # Test accuracy
    test(net, testloader)
    # Save the trained model parameters
    torch.save(net.state_dict(), save_path)



