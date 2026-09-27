import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import numpy as np
from function import test_val_dataset, train_val_dataset, train, validate, ApplyTransformSubset, build_model
import random
import csv
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--model", required=True, 
                    choices=['ResNet18','EfficientNetB0','ConvNeXtTiny','SwinTiny'],
                    help="Architectures supported:\n- ResNet18\n- EfficientNetB0\n- ConvNeXtTiny\n- SwinTiny ")
args = parser.parse_args()

if args.model == 'ResNet18':
    from configs.ResNet18 import config
elif args.model == 'EfficientNetB0':
    from configs.EfficientNetB0 import config
elif args.model == 'ConvNeXtTiny':
    from configs.ConvNeXtTiny import config
elif args.model == 'SwinTiny':
    from configs.SwinTiny import config
    
print(args.model)

# the directory for the data
data_dir = r'./kvasir-dataset-v2'
random_state = 42

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

# image transformations coresponding to each model
augmentation = transforms.Compose([
    transforms.Resize(config['image_resize']),
    transforms.CenterCrop(config['image_center_crop']),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p = 0.5),
    transforms.RandomApply([transforms.RandomRotation(degrees=30)], p=0.7),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25)], p = 0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std = config['std'])
])

# image augmentation for training
eval_transform = transforms.Compose([
    transforms.Resize(config['image_resize']),
    transforms.CenterCrop(config['image_center_crop']),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std = config['std'])
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

dataloader = {'train':DataLoader(final_datasets['train'], batch_size=config['batch_size'], shuffle=True),
              'val':DataLoader(final_datasets['val'], batch_size=config['batch_size'], shuffle=False),
              'test':DataLoader(final_datasets['test'], batch_size=config['batch_size'], shuffle=False)
}

# iters through the directories to get the label
images, labels = next(iter(dataloader['train']))
print(images.shape, labels.shape)
print(labels)

# loading the model
model = build_model(config)

# setting the right device
model = model.to(device)

# defining the loss function and optimizer
criterion = nn.CrossEntropyLoss() # loss in classification
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate']) # optimizer in training
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'])

# training loop
best_acc = 0
max_epochs = 10
nr_epochs = 0

# writing the values to a csv
fields = ['validation loss', 'validation accuracy', 'train loss', 'train accuracy']
filename = 'csv/' + config['model_name'] + '.csv'

with open(filename, 'w') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)

    for epoch in range(config['epochs']):
        print(f"*******Epoch {epoch + 1}\n********")
        row = []
        train_loss, train_acc = train(dataloader['train'], model, device, criterion, optimizer)
        loss, acc = validate(dataloader['val'], model, device, criterion)
        scheduler.step()
        
        # updating the row
        row.append(loss)
        row.append(acc)
        row.append(train_loss)
        row.append(train_acc)
        csvwriter.writerow(row)

        if acc > best_acc:
            best_acc = acc
            nr_epochs = 0
            name = 'models/best_model_' + config['model_name'] + '.pth'
            torch.save(model.state_dict(), name)
        else:
            nr_epochs += 1

        if nr_epochs >= max_epochs:
            print(f"Stopped at epoch {epoch+1}")
            break
        