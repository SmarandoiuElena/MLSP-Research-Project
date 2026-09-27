import matplotlib.pyplot as plt 
import numpy as np
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

filename = "csv/" + config['model_name'] + '.csv'

val_loss_over_epochs = []
val_acc_over_epochs = []
train_loss_over_epochs = []
train_acc_over_epochs = []

with open(filename, 'r') as csvfile:
    data = csv.reader(csvfile)
    i = 0
    for row in data:
        if i != 0:
            val_loss_over_epochs.append(float(row[0]))
            val_acc_over_epochs.append(float(row[1]))
            train_loss_over_epochs.append(float(row[2]))
            train_acc_over_epochs.append(float(row[3]))
        i += 1
        
plt.figure()
y1 = np.array(val_loss_over_epochs)
y2 = np.array(train_loss_over_epochs)
x = np.array([i for i in range(1, len(val_loss_over_epochs) + 1)])
plt.plot(x, y1, color='r', label='validation loss')
plt.plot(x, y2, color='b', label='train loss')
plt.title("Loss_over_epochs_" + config['model_name'] )
plt.legend()
plt.savefig("metrics/Loss_over_epochs_" + config['model_name'] + ".jpg")
plt.show()

plt.figure()
y1 = np.array(val_acc_over_epochs)
y2 = np.array(train_acc_over_epochs)
x = np.array([i for i in range(1, len(val_acc_over_epochs) + 1)])
plt.plot(x, y1, color='r', label='validation accuracy')
plt.plot(x, y2, color='b', label='train accuracy')
plt.title("Avg_accuracy_over_epochs_" + config['model_name'])
plt.legend()
plt.savefig("metrics/Avg_accuracy_over_epochs_" + config['model_name']+ ".jpg")
plt.show()
