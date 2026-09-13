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
augmentation to the training set to prevent overfitting. 

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
https://www.mdpi.com/1424-8220/23/6/3176 (for pipeline and augmentation)
https://www.nature.com/articles/s41598-024-53955-8 (suppoorting augmentation decision)

