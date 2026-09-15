import torch
from torch.utils.data import Subset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

class ApplyTransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)
    
def train_val_dataset(dataset, val_split = 0.30, random = 42):
    targets = dataset.targets
    
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, stratify=targets, random_state=random)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    
    return datasets

def test_val_dataset(dataset, val_split = 0.50, random = 42):
    val_targets =  [dataset['val'].dataset.targets[i] for i in dataset['val'].indices]
    
    test_idx, val_idx = train_test_split(list(range(len(dataset['val']))), test_size=val_split, stratify=val_targets, random_state=random)
    datasets2 = {}
    datasets2['train'] = dataset['train']
    datasets2['val'] = Subset(dataset['val'], val_idx)
    datasets2['test'] = Subset(dataset['val'], test_idx)
    
    return datasets2

def train(dataloader, model, device, loss_fn, optimizer):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    
    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # the prediction error
        prediction = model(images)
        loss = loss_fn(prediction, labels)
        
        train_loss += loss_fn(prediction, labels).item()
        correct += (prediction.argmax(1) == labels).type(torch.float).sum().item()
        
        # backprog
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 20 == 0:
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
    train_acc = correct / size
    train_loss /= num_batches
        
    return train_loss, train_acc
    
def test(dataloader, model, device, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    test_loss, correct = 0, 0
    predict = []
    true = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            
            predict.extend(pred.argmax(1).cpu().numpy())
            true.extend(labels.cpu().numpy())
            
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
            
    classes = ('dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
               'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis')
    test_loss /= num_batches
    correct /= size
    
    cf_matrix = confusion_matrix(true, predict)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=classes,
        columns=classes
    )
    
    plt.figure(figsize=(10, 8))
    
    sn.heatmap(df_cm,
               annot = True,
               fmt=".2f",
               cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
    plt.tight_layout()
    plt.savefig("Confusion_matrix_resnet18.pdf")
    plt.show()
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    report = classification_report(true, predict, target_names=classes, digits=4)
    print(report)
    
    return correct, test_loss
    
def validate(dataloader, model, device, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    total_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            
            total_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
            
    total_loss /= num_batches
    correct /= size
    
    print(f"Total Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {total_loss:>8f} \n")      
    return total_loss, correct
