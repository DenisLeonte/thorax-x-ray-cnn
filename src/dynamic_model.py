import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicCNN(nn.Module):
    def __init__(self, config, num_classes=14):
        """
        config structure example:
        {
            "blocks": [
                {"filters": 32, "kernel": 3, "pool": True},
                {"filters": 64, "kernel": 3, "pool": True}
            ],
            "fc_layers": [512],
            "dropout": 0.5
        }
        """
        super(DynamicCNN, self).__init__()
        self.config = config
        
        # Build Convolutional Blocks
        self.features = nn.Sequential()
        in_channels = 3
        
        # Store output size calculation (assuming 224x224 input)
        current_size = 224
        
        for i, block in enumerate(config['blocks']):
            out_channels = block['filters']
            kernel_size = block['kernel']
            use_pool = block['pool']
            padding = kernel_size // 2
            
            self.features.add_module(f"conv{i}", nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            self.features.add_module(f"bn{i}", nn.BatchNorm2d(out_channels))
            self.features.add_module(f"relu{i}", nn.ReLU(inplace=True))
            
            if use_pool:
                self.features.add_module(f"pool{i}", nn.MaxPool2d(2, 2))
                current_size = current_size // 2
                
            in_channels = out_channels
            
        self.last_conv_channels = in_channels
        self.last_spatial_size = current_size
        
        # Adaptive pooling to handle variable spatial sizes gracefully
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Build Classifier
        self.classifier = nn.Sequential()
        in_features = self.last_conv_channels
        
        for i, units in enumerate(config['fc_layers']):
            self.classifier.add_module(f"fc{i}", nn.Linear(in_features, units))
            self.classifier.add_module(f"fc_relu{i}", nn.ReLU(inplace=True))
            self.classifier.add_module(f"drop{i}", nn.Dropout(config['dropout']))
            in_features = units
            
        self.classifier.add_module("output", nn.Linear(in_features, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
