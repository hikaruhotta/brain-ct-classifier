import torch
import torch.nn as nn

from classifier.resnet3d import r3d_18, mc3_18, r2plus1d_18

class Classifier3D(torch.nn.Module):
    
    def __init__(self, args):
        super().__init__()
        # Initialize 3D Resnet Model
        self.resnet3d = None
        if args.resnet3d_model == "r3d_18":
            self.resnet3d = r3d_18(pretrained=True)  # 18 layer Resnet3D
        elif args.resnet3d_model == "mc3_18":
            self.resnet3d = mc3_18(pretrained=True)  # 18 layer Mixed Convolution network
        else:
            self.resnet3d = r2plus1d_18(pretrained=True)  # 18 layer deep R(2+1)D network
        
        self.resnet3d_out_features = self.resnet3d.fc.out_features
        
        self.last_ =  nn.Linear(self.resnet3d_out_features, args.num_classes)
        
        
    def forward(self, x):
        x = self.resnet3d(x)
        x = self.last_(x)
        return x
