import numpy as np
import torch
from torchvision import datasets, transforms
from collections import defaultdict

def train(model, train_loader, optimizer, criterion, epochs=10, val_loader=None, 
          verbose=False, val_verbose=True, device="cpu", regression=False, val_acts = False):
    model.train()
    val_losses = []
    val_accs = []
    if val_acts:
        acts = defaultdict(list)
    else:
        acts = None
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            if regression:
                output = output.flatten()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0 and verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        if val_loader is not None:
            if val_verbose: 
                print("Validation: ", end = "")
            loss, acc = test(model, val_loader, criterion, device=device, regression=regression, verbose=val_verbose)
            val_losses.append(loss)
            val_accs.append(acc)
            if val_acts:
                for key, value in model.activations.items():
                    acts[key].append(value.cpu())
    return val_losses, val_accs, acts

def test(model, test_loader, criterion, device="cpu", regression=False, verbose=True, metrics=False):
    model.eval()
    if metrics and hasattr(model, "activations"):
        for key, value in model.activations.items():
            model.activations[key] = torch.Tensor().to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if regression:
                output = output.flatten()
            test_loss += criterion(output, target).item() # sum up batch loss
            if not regression:
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    
    if verbose:
        print('Average loss: {:.4f}'.format(test_loss), end="")
        if not regression:
            print(', Accuracy: {}/{} ({:.2f}%)'.format(correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)), end="")
        if not metrics:
            print()
    
    return test_loss, correct/len(test_loader.dataset)

"""
Create a class for toy datasets
"""
class ToyData(torch.utils.data.Dataset):
    def __init__(self, input_dim, signal_dim, num_samples=1000, regression=False, masked=False):
        self.X_base = torch.randn(num_samples, input_dim)
        self.multiplier = torch.randn(input_dim)
        self.multiplier = torch.cat((self.multiplier, self.multiplier), dim=0)
        self.input_dim = input_dim
        self.signal_dim = signal_dim
        self.window_index = 0
        self.regression = regression
        self.masked = masked
        self.compute_X()
        self.compute_y()

    def __len__(self):
        return len(self.X_base)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def compute_X(self):
        self.X = self.X_base.clone()
        if self.masked:
            if self.window_index+self.signal_dim <= self.X.shape[1]:
                self.X[:,0:self.window_index] = 0
                self.X[:,self.window_index+self.signal_dim:] = 0
            else:
                self.X[:,self.window_index+self.signal_dim-self.X.shape[1]:self.window_index] = 0

    def compute_y(self):
        stop = self.window_index+self.signal_dim
        y = torch.matmul(torch.cat((self.X, self.X), dim=1)[:,self.window_index:stop], self.multiplier[self.window_index:stop])
        if not self.regression:
            y = y > 0
            self.y = y.long()
        else:
            self.y = y
        
    def shift_window(self, shift):
        self.window_index += shift
        if self.window_index > self.X_base.shape[1]:
            self.window_index -= self.X_base.shape[1]
        self.compute_X()
        self.compute_y()

    def shift_distribution(self, multiplier=1.0, adder=0.0, recompute_y=True):
        self.X_base = self.X_base * multiplier + adder
        self.compute_X()
        if recompute_y:
            self.compute_y()


def mnist_dataset():
    dataset = datasets.MNIST('../data', train=True, download=True,
                     transform=transforms.Compose([ 
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    return dataset

"""
Turn a dataset into train, val, and test split dataloaders
"""
def split_dataset(X, y=None, X_test=None, y_test=None, val_size=0.1, test_size=0.1, batch_size=128, shuffle_val=False):
    if y is not None:
        dataset = torch.utils.data.TensorDataset(X, y)
    else:
        dataset = X
    if X_test is None:
        dataset, test_set = torch.utils.data.random_split(dataset, lengths=[int((1-test_size)*len(dataset)), int(test_size*len(dataset))])
    elif y_test is not None:
        test_set = torch.utils.data.TensorDataset(X_test, y_test)
    else:
        test_set = X_test
    train_set, val_set = torch.utils.data.random_split(dataset, lengths=[int((1-val_size)*len(dataset)), int(val_size*len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle_val)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def get_swiss_rolls(n_samples=1000, noise_factor=0.9, tasks = 10):

    t = np.sort(1.5 * np.pi * (1 + 2*np.random.uniform(size=n_samples)))
    t2 = np.sort(1.5 * np.pi * (2/3 + 2*np.random.uniform(size=n_samples)))

    x = t * np.cos(t)
    y = t * np.sin(t)

    x2 = t2 * np.cos(t2+np.pi/2)
    y2 = t2 * np.sin(t2+np.pi/2)

    X = np.vstack((x, y)).T
    noise = noise_factor * np.random.random(size=(n_samples, 2))
    X += noise
    t = np.squeeze(t)

    X2 = np.vstack((x2, y2)).T
    noise2 = noise_factor * np.random.random(size=(n_samples, 2))
    X2 += noise2
    t2 = np.squeeze(t2)

    k = tasks
    X_ = np.array_split(X, k)
    X2_ = np.array_split(X2, k)

    data = [np.concatenate((X_[i], X2_[i])) for i in range(k)]
    labels = [np.concatenate((np.zeros(len(X_[i])), np.ones(len(X2_[i])))).astype(int) for i in range(k)]

    dataset = np.concatenate((X,X2))
    xmin, xmax = dataset[:, 0].min() - 1, dataset[:, 0].max() + 1
    ymin, ymax = dataset[:, 1].min() - 1, dataset[:, 1].max() + 1
    steps = 100
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)
    grid = torch.autograd.Variable(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())

    datasets = [split_dataset(torch.Tensor(data[i]), torch.Tensor(labels[i]).long(), val_size = 0.1, test_size = 0.1, batch_size = 16) for i in range(k)]
