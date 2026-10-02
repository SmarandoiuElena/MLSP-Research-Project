import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from function import test_val_dataset, train_val_dataset, test, ApplyTransformSubset, build_model
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
    
# the device used for computation
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

data_dir = r'./kvasir-dataset-v2'

num_classes = config['nr_classes']
batch_size = config['batch_size']
random_state = 42

eval_transform = transforms.Compose([
    transforms.Resize((config['image_resize'], config['image_resize'])),
 #   transforms.CenterCrop(config['image_center_crop']),
    transforms.ToTensor(),
    transforms.Normalize(mean=config['mean'], std=config['std'])
])

# spliting the set into test/train/val
my_dataset = datasets.ImageFolder(data_dir)
my_datasets = train_val_dataset(my_dataset, 0.3, random_state)
my_datasets = test_val_dataset(my_datasets, 0.5, random_state)

final_dataset = ApplyTransformSubset(my_datasets['test'], eval_transform)
final_loader = DataLoader(final_dataset, batch_size=batch_size, shuffle=False)

# loading the model
model = build_model(config, pretrained = final_dataset)
model.load_state_dict(torch.load('models/best_model_'+ config['model_name'] + '.pth', map_location=device))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
test(final_loader, model, device, criterion, config)
