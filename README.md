# MLSP Project - Medical images classification 
## Smarandoiu Elena-Andrada

### First Pipepline(experimental)

The first ever pipeline I tried used the ResNet-18 model with pretrained weights.

Parameters:

image_size = 255 (left by accident and forgot to be modified)
num_classes = 8
epochs = 10
batch_size=32
loss_fn = CrossEntropyLoss
optimizer = Adam with lr = 0.001
My default train/test/validate subsets are 70%/15%/15%.
Trained on my CPU.

After Epoch 3 the model had an imbalance and the accuracy dropped. That may be because
I used a constant learing rate of 0.001 that is too fast. I also did not include image
augmentation to the training set so the model strated overfitting. 

Result:

```bash
*******Epoch 1
********
loss: 2.127033 [   32/ 5600]
loss: 0.362255 [ 3232/ 5600]
Total Error: 
 Accuracy: 69.7%, Avg loss: 0.950311 

*******Epoch 2
********
loss: 0.343875 [   32/ 5600]
loss: 0.373111 [ 3232/ 5600]
Total Error: 
 Accuracy: 85.5%, Avg loss: 0.355746 

*******Epoch 3
********
loss: 0.356200 [   32/ 5600]
loss: 0.454971 [ 3232/ 5600]
Total Error: 
 Accuracy: 77.1%, Avg loss: 0.922786 

*******Epoch 4
********
loss: 0.321209 [   32/ 5600]
loss: 0.356717 [ 3232/ 5600]
Total Error: 
 Accuracy: 73.2%, Avg loss: 0.761545 

*******Epoch 5
********
loss: 0.226473 [   32/ 5600]
loss: 0.157488 [ 3232/ 5600]
Total Error: 
 Accuracy: 77.8%, Avg loss: 0.694455 

*******Epoch 6
********
loss: 0.347281 [   32/ 5600]
loss: 0.148285 [ 3232/ 5600]
Total Error: 
 Accuracy: 80.7%, Avg loss: 0.483458 

*******Epoch 7
********
loss: 0.348611 [   32/ 5600]
loss: 0.184555 [ 3232/ 5600]
Total Error: 
 Accuracy: 81.3%, Avg loss: 0.610616 

*******Epoch 8
********
loss: 0.084579 [   32/ 5600]
loss: 0.075264 [ 3232/ 5600]
Total Error: 
 Accuracy: 86.7%, Avg loss: 0.415986 

*******Epoch 9
********
loss: 0.034485 [   32/ 5600]
loss: 0.034047 [ 3232/ 5600]
Total Error: 
 Accuracy: 84.6%, Avg loss: 0.558827 

*******Epoch 10
********
loss: 0.052264 [   32/ 5600]
loss: 0.165139 [ 3232/ 5600]
Total Error: 
 Accuracy: 86.3%, Avg loss: 0.374461 

Test Error: 
 Accuracy: 86.8%, Avg loss: 0.391537 
```

### Second Pipeline

Papers:
https://www.mdpi.com/1424-8220/23/6/3176 (for pipeline and augmentation) [1]
https://www.nature.com/articles/s41598-024-53955-8 (suppoorting augmentation decision) [2]

I started implementing image augmentation for which I took inspiration from two paper, both
focusing on the classification of endoscopic images. The first paper concluded that high changes
in hue and color don't save the model from overfitting when it comes to medical images (even 
deleting the two classes with coloured polyps from the dataset). It proposed 
a change in only brightness and contrast along with slight rotation so that the change overall
would be more subtle.

The second paper's findings supported the idea that only subtle changes should be made in colour
and in addition also varried the contrast of the images. It also argued that geometrical variations
such as horizontal or vertical flips are safe for medical images because they do not have a definite up, down or direction.

For the augmentation process I combined these findings and set a probability for each operation.

I used the way paper [1] dealt with the imbalance after epoch 3 and used a StepLr scheduler. I kept
the intial learning rate for the Adam as 0.001 but decreased it at every 2 steps. My training loop
keeps track of the best accuracy, saves and updates the best model. I initially put a maximum of
20 epochs because my model starting downgrading around the epoch 15-16, and had a tolerance of of
5 epochs (exiting early if the model did not improve). I chose step = 2 to keep the ration in the first paper (that downgraded the lr every 10 steps for 100 epochs). By lowering the step I obtained
a much stable and nicer result.

```bash
*******Epoch 1
********
loss: 2.080967 [   64/ 5600]
Total Error: 
 Accuracy: 79.4%, Avg loss: 0.527032 

*******Epoch 2
********
loss: 0.285388 [   64/ 5600]
Total Error: 
 Accuracy: 77.4%, Avg loss: 0.578005 

*******Epoch 3
********
loss: 0.473114 [   64/ 5600]
Total Error: 
 Accuracy: 82.8%, Avg loss: 0.457399 

*******Epoch 4
********
loss: 0.096653 [   64/ 5600]
Total Error: 
 Accuracy: 84.5%, Avg loss: 0.401938 

*******Epoch 5
********
loss: 0.189172 [   64/ 5600]
Total Error: 
 Accuracy: 88.8%, Avg loss: 0.296597 

*******Epoch 6
********
loss: 0.232901 [   64/ 5600]
Total Error: 
 Accuracy: 85.2%, Avg loss: 0.336027 

*******Epoch 7
********
loss: 0.233013 [   64/ 5600]
Total Error: 
 Accuracy: 90.1%, Avg loss: 0.279292 

*******Epoch 8
********
loss: 0.116723 [   64/ 5600]
Total Error: 
 Accuracy: 85.7%, Avg loss: 0.449767 

*******Epoch 9
********
loss: 0.132275 [   64/ 5600]
Total Error: 
 Accuracy: 90.7%, Avg loss: 0.261461 

*******Epoch 10
********
loss: 0.140527 [   64/ 5600]
Total Error: 
 Accuracy: 91.1%, Avg loss: 0.256072 

*******Epoch 11
********
loss: 0.136913 [   64/ 5600]
Total Error: 
 Accuracy: 89.3%, Avg loss: 0.303025 

*******Epoch 12
********
loss: 0.147198 [   64/ 5600]
Total Error: 
 Accuracy: 91.0%, Avg loss: 0.260764 

*******Epoch 13
********
loss: 0.167088 [   64/ 5600]
Total Error: 
 Accuracy: 90.1%, Avg loss: 0.307299 

*******Epoch 14
********
loss: 0.157253 [   64/ 5600]
Total Error: 
 Accuracy: 91.8%, Avg loss: 0.251725 

*******Epoch 15
********
loss: 0.170722 [   64/ 5600]
Total Error: 
 Accuracy: 90.6%, Avg loss: 0.294685 

*******Epoch 16
********
loss: 0.098334 [   64/ 5600]
Total Error: 
 Accuracy: 89.5%, Avg loss: 0.304484 

*******Epoch 17
********
loss: 0.104244 [   64/ 5600]
Total Error: 
 Accuracy: 91.0%, Avg loss: 0.248873 

*******Epoch 18
********
loss: 0.152483 [   64/ 5600]
Total Error: 
 Accuracy: 91.7%, Avg loss: 0.260493 

*******Epoch 19
********
loss: 0.106332 [   64/ 5600]
Total Error: 
 Accuracy: 91.5%, Avg loss: 0.276151 

Stopped at epoch 19
Using cuda device Test Error:   Accuracy: 90.1%, Avg loss: 0.252590
```
#### Metrics for step = 10
[Average accuracy over epochs with step = 10](./metrics/Avg_accuracy__over_epochs-1_page-0001.jpg)
[Loss over epochs with step = 10](./metrics/Loss_over_epochs-1_page-0001.jpg)

#### Metrics for step = 2
[Average accuracy over epochs with step = 2](./metrics/Avg_accuracy__over_epochs-2_page-0001.jpg)
[Loss over epochs with step = 2](./metrics/Loss_over_epochs-2_page-0001.jpg)

