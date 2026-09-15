import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets, models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from function import test_val_dataset, train_val_dataset, train, test, validate, ApplyTransformSubset
import random

# the directory for the data
data_dir = r'./kvasir-dataset-v2/kvasir-dataset-v2'

image_size = 224
num_classes = 8
epochs = 50
batch_size = 64
learning_rate = 0.001
random_state = 42
step_size = 2
gamma = 0.9

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(random_state)

# the device used for computation
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# the pretrained model architecture
model = models.resnet18(models.ResNet18_Weights.DEFAULT)

# image transformations coresponding to each model
augmentation = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p = 0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.7),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)], p = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# image augmentation for training
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# spliting the set into test/train/val
my_dataset = datasets.ImageFolder(data_dir)
my_datasets = train_val_dataset(my_dataset, 0.3, random_state)
my_datasets = test_val_dataset(my_datasets, 0.5, random_state)

print(len(my_datasets['train']))
print(len(my_datasets['val']))
print(len(my_datasets['test']))

# applying augmentation for training images
final_datasets = {
    'train': ApplyTransformSubset(my_datasets['train'], augmentation),
    'val': ApplyTransformSubset(my_datasets['val'], eval_transform),
    'test': ApplyTransformSubset(my_datasets['test'], eval_transform)
}

dataloader = {'train':DataLoader(final_datasets['train'], batch_size=batch_size, shuffle=True),
              'val':DataLoader(final_datasets['val'], batch_size=batch_size, shuffle=False),
              'test':DataLoader(final_datasets['test'], batch_size=batch_size, shuffle=False)}

# iters through the directories to get the label
images, labels = next(iter(dataloader['train']))
print(images.shape, labels.shape)
print(labels)

# setting the right number of classes and the device
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# defining the loss function and optimizer
criterion = nn.CrossEntropyLoss() # loss in classification
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # optimizer in training
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# training loop
best_acc = 0
max_epochs = 10
nr_epochs = 0

val_loss_over_epochs = []
val_acc_over_epochs = []

train_loss_over_epochs = []
train_acc_over_epochs = []

for epoch in range(epochs):
    print(f"*******Epoch {epoch + 1}\n********")
    train_loss, train_acc = train(dataloader['train'], model, device, criterion, optimizer)
    loss, acc = validate(dataloader['val'], model, device, criterion)

    val_loss_over_epochs.append(loss)
    val_acc_over_epochs.append(acc)
    train_acc_over_epochs.append(train_acc)
    train_loss_over_epochs.append(train_loss)

    scheduler.step()

    if acc > best_acc:
        best_acc = acc
        nr_epochs = 0
        torch.save(model.state_dict(), 'best_model_resnet18.pth')
    else:
        nr_epochs += 1

    if nr_epochs >= max_epochs:
        print(f"Stopped at epoch {epoch+1}")
        break

y = np.array(val_loss_over_epochs)
x = np.array([i for i in range(1, len(val_loss_over_epochs) + 1)])
plt.plot(x, y)
plt.title("Validation_Loss_over_epochs")
plt.savefig("Validation_Loss_over_epochs.pdf")
plt.show()

y = np.array(val_acc_over_epochs)
x = np.array([i for i in range(1, len(val_acc_over_epochs) + 1)])
plt.plot(x, y)
plt.title("Validation_Avg_accuracy_over_epochs")
plt.savefig("Validation_Avg_accuracy_over_epochs.pdf")
plt.show()

y = np.array(train_loss_over_epochs)
x = np.array([i for i in range(1, len(train_loss_over_epochs) + 1)])
plt.plot(x, y)
plt.title("Train_loss_over_epochs")
plt.savefig("Train_loss_over_epochs.pdf")
plt.show()

y = np.array(train_acc_over_epochs)
x = np.array([i for i in range(1, len(train_acc_over_epochs) + 1)])
plt.plot(x, y)
plt.title("Train_Avg_accuracy_over_epochs")
plt.savefig("Train_Avg_accuracy_over_epochs.pdf")
plt.show()
