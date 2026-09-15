# the script for testing
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from function import test_val_dataset, train_val_dataset, test, ApplyTransformSubset

# the device used for computation
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

data_dir = r'./kvasir-dataset-v2/kvasir-dataset-v2'

num_classes = 8
batch_size = 64
random_state = 42

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

final_dataset = ApplyTransformSubset(my_datasets['test'], eval_transform)
final_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=False)

# loading the model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('best_model_resnet18.pth', map_location=device))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
acc, loss = test(final_loader, model, device, criterion)
