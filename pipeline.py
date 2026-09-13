import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms, datasets, models
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def train_val_dataset(dataset, val_split = 0.30):
    targets = dataset.targets
    
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split, stratify=targets)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    
    return datasets

def test_val_dataset(dataset, val_split = 0.50):
    val_targets =  [dataset['val'].dataset.targets[i] for i in dataset['val'].indices]
    
    test_idx, val_idx = train_test_split(list(range(len(dataset['val']))), test_size=val_split, stratify=val_targets)
    datasets2 = {}
    datasets2['train'] = dataset['train']
    datasets2['val'] = Subset(dataset['val'], val_idx)
    datasets2['test'] = Subset(dataset['val'], test_idx)
    
    return datasets2

def train(dataloader, model, device, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        
        # the prediction error
        prediction = model(images)
        loss = loss_fn(prediction, labels)
        
        # backprog
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(images)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test(dataloader, model, device, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    test_loss, correct = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            test_loss += loss_fn(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
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
    
    
# the device used for computation
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# the pretrained model architecture
model = models.resnet18(models.ResNet18_Weights.DEFAULT)

# the directory for the data
data_dir = r'./kvasir-dataset-v2'
image_size = 255
num_classes = 8
epochs = 10

# image transformations coresponding to each model
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor()
   # transforms.Rotate(limit=30, p=0.7),
   # transforms.RandomBrightnessContrats(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
])

dataset = datasets.ImageFolder(data_dir, transform=transform)

my_datasets = train_val_dataset(dataset)
my_datasets = test_val_dataset(my_datasets)

print(len(my_datasets['train']))
print(len(my_datasets['val']))
print(len(my_datasets['test']))

dataloader = {x:DataLoader(my_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val', 'test']}

# iters through the directories to get the label
images, labels = next(iter(dataloader['train']))
print(images.shape, labels.shape)

labels = dataset.classes

print(labels)

# setting the right number of classes and the device
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# defining the loss function and optimizer
criterion = nn.CrossEntropyLoss() # loss in classification
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optimizer in training

# training loop
for epoch in range(epochs):
    
    print(f"*******Epoch {epoch + 1}\n********")
    train(dataloader['train'], model, device, criterion, optimizer)
    validate(dataloader['val'], model, device, criterion)
    
# testing
test(dataloader['test'], model, device, criterion)
    
    