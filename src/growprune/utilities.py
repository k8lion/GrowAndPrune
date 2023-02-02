import numpy as np
import torch
from torchvision import datasets, transforms
from collections import defaultdict
import math
import h5py
import os

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
                if len(output.shape) == 2:
                    pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                else:
                    pred = output > 0
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

class TransferToyData(torch.utils.data.Dataset):
    def __init__(self, angle: float = 0., x: float = 0., y: float = 0., line: bool = True, num_samples: int = 1000):
        self.angle = angle
        self.line = line
        #self.X = 2*torch.rand(num_samples, 2) - 1
        self.X = torch.randn(num_samples, 2)
        if self.line:
            self.X[:,0] = 0
        self.y = (self.X[:,1] > 0).float()
        self.X = self.X @ torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        self.X[:,0] += x
        self.X[:,1] += y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EffDimToyData(torch.utils.data.Dataset):
    def __init__(self, features: int = 64, effdim: int = 32, outdim: int = -1, num_samples: int = 1000):
        self.features = features
        self.effdim = effdim
        self.outdim = outdim
        self.X = torch.randn(num_samples, features)
        if effdim < features:
            self.X[:,effdim:] = self.X[:,:effdim]@torch.randn(effdim, features-effdim)
        response = torch.sum(self.X[:,outdim] if outdim >= 0 else self.X, dim=1)
        self.y = (response > torch.sum(response)/len(response)).float()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
    train_set, val_set = torch.utils.data.random_split(dataset, lengths=[int((1-(val_size/(1-test_size)))*len(dataset)), int((val_size/(1-test_size))*len(dataset))])
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


class Galaxy10Dataset(torch.utils.data.Dataset):
    def __init__(self, path_to_dir='~', transform=transforms.ToTensor(), mode="train"):
        if mode == "train":
            file = str(path_to_dir)+"/data/Galaxy10_DECals_trainval.h5"
        else:
            file = str(path_to_dir)+"/data/Galaxy10_DECals_test.h5"
        
        f = h5py.File(file, 'r')
        labels, images = f['labels'], f['images']

        self.x = images
        self.y = torch.from_numpy(labels[:]).long()
        self.num_samples = len(images)

        self.transform = transform

    def __getitem__(self, item):
        img = self.x[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.y[item]

    def __len__(self):
        return self.num_samples

def get_galaxy10_dataloaders(path_to_dir = "~", validation_split=0.1, batch_size=32, small=True):
    if batch_size < 0:
        batch_size = 32 if small else 16
    print(os.path.exists(str(path_to_dir)+"/data/Galaxy10_DECals_trainval.h5"), os.path.exists(str(path_to_dir)+"/data/Galaxy10_DECals.h5"))
    if not os.path.exists(str(path_to_dir)+"/data/Galaxy10_DECals_trainval.h5"):
        if os.path.exists(str(path_to_dir)+"/data/Galaxy10_DECals.h5"):
            print("making dataset")
            make_galaxy10_traintest(path_to_dir)
        else:
            print("No data found")
            return None, None, None
    totensor = transforms.ToTensor()
    if small:
        transform = transforms.Compose([
            totensor,
            transforms.Resize(64),
            transforms.Normalize([0.16733793914318085, 0.16257789731025696, 0.1588301658630371], [0.1201716959476471, 0.11228285729885101, 0.10515376180410385]),
        ])
    else:
        transform = transforms.Compose([
            totensor,
            transforms.Normalize([0.16683201, 0.16196689, 0.15829432], [0.12819551, 0.11757845, 0.11118137]),
        ])
    galaxy10_train = Galaxy10Dataset(mode='train', transform=transform, path_to_dir=path_to_dir)
    shuffle_dataset = True
    random_seed = 42
    dataset_size = galaxy10_train.num_samples
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(galaxy10_train, batch_size=batch_size, 
                                            sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(galaxy10_train, batch_size=batch_size,
                                                    sampler=valid_sampler)

    galaxy10_test = Galaxy10Dataset(mode='test', transform=transform, path_to_dir=path_to_dir)
    test_loader = torch.utils.data.DataLoader(galaxy10_test, batch_size=batch_size)

    return train_loader, validation_loader, test_loader


def make_galaxy10_traintest(path_to_dir = "..", test_split=0.1, seed=42):
    np.random.seed(seed)
    file = str(path_to_dir)+"/data/Galaxy10_DECals.h5"
    f = h5py.File(file, 'r')
    print(f.keys())
    labels, images = f['ans'], f['images']
    inds = np.arange(len(labels))
    np.random.shuffle(inds)
    split_ind = int(np.floor(test_split * len(inds)))
    tv_inds, test_inds = sorted(inds[split_ind:]), sorted(inds[:split_ind])
    tv_f = h5py.File(str(path_to_dir)+"/data/Galaxy10_DECals_trainval.h5", 'w')
    tv_f.create_dataset('images', data=images[tv_inds,:,:,:])
    tv_f.create_dataset('labels', data=labels[tv_inds])
    tv_f.close()
    test_f = h5py.File(str(path_to_dir)+"/data/Galaxy10_DECals_test.h5", 'w')
    test_f.create_dataset('images', data=images[test_inds,:,:,:])
    test_f.create_dataset('labels', data=labels[test_inds])
    test_f.close()