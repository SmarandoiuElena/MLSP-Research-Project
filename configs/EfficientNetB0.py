config = {
    'model_name': 'EfficientNetB0',
    'nr_classes': 8,
    'image_resize': 224,
    'image_center_crop': 224,
    'epochs': 30,
    'batch_size': 64,
    'learning_rate': 0.001,
    'momentum': 0.9,
    'weight_decay': 0,
    'step_size': 2,
    'gamma': 0.9,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'classes': ('dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
                'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis')
}