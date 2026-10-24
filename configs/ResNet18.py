config = {
    'model_name': 'ResNet18',
    'nr_classes': 8,
    'image_resize': 224,
    'image_center_crop': 224,
    'epochs': 30,
    'batch_size': 64,
    'learning_rate_backbone': 3e-5,
    'learning_rate_head': 1e-3,
    'weight_decay': 0.5,
    'gamma': 0.9,
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'classes': ('dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 'normal-cecum',
                'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis')
}