import torch
from torchvision import datasets, transforms

def train(model, train_loader, optimizer, criterion, epochs=10, val_loader=None, verbose=True, device="cpu", regression=False):
    model.train()
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
            print("Validation: ", end = "")
            test(model, val_loader, criterion, device=device, regression=regression)

def test(model, test_loader, criterion, device="cpu", regression=False):
    model.eval()
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
    
    if regression:
        print('Average loss: {:.4f}'.format(test_loss))
    else:
        print('Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

"""
Create a class for toy datasets
"""
class ToyData(torch.utils.data.Dataset):
    def __init__(self, input_dim, signal_dim, num_samples=1000, regression=False):
        self.X = torch.randn(num_samples, input_dim)
        self.multiplier = torch.randn(input_dim)
        self.multiplier = torch.cat((self.multiplier, self.multiplier), dim=0)
        self.input_dim = input_dim
        self.signal_dim = signal_dim
        self.window_index = 0
        self.regression = regression
        self.compute_y()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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
        if self.window_index > self.X.shape[1]:
            self.window_index = 0
        self.compute_y()

    def shift_distribution(self, multiplier=1.0, adder=0.0):
        self.X = self.X * multiplier + adder
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
def split_dataset(X, y=None, X_test=None, y_test=None, val_size=0.1, test_size=0.1, batch_size=128):
    if y is not None:
        dataset = torch.utils.data.TensorDataset(X, y)
    else:
        dataset = X
    if X_test is None:
        dataset, test_set = torch.utils.data.random_split(dataset, lengths=[int((1-test_size)*len(dataset)), int(test_size*len(dataset))])
    elif y_test is not None:
        test_set = torch.utils.data.TensorDataset(X_test, y_test)
    else:
        test_ste = X_test
    train_set, val_set = torch.utils.data.random_split(dataset, lengths=[int((1-val_size)*len(dataset)), int(val_size*len(dataset))])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader