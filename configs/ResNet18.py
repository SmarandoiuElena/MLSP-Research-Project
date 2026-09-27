config = {
    'model_name': 'ResNet18',
    'nr_classes': 8,
    'image_resize': 256,
    'image_center_crop': 224,
    'epochs': 50,
    'batch_size': 64,
    'learning_rate': 0.001,
    'step_size': 2,
    'gamma': 0.9,
    'mean':[0.485, 0.456, 0.406],
    'std':[0.229, 0.224, 0.225],
    'classes': ('dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
               'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis')
}