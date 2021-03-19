import torch
import torch.nn as nn
import torch.nn.functional as F

# from classifier.resnet3d import r3d_18, mc3_18, r2plus1d_18
from args.multi_task_classifier_3d_train_arg_parser import MTClassifier3DTrainArgParser

from torchvision.models.video import r3d_18, mc3_18, r2plus1d_18


class ClassifierHead(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.x1 = nn.Linear(in_features, in_features // 2)
        nn.init.kaiming_normal_(self.x1.weight)
        self.dropout1 = nn.Dropout(p=0.2)
        self.x2 = nn.Linear(in_features // 2, out_features)
        nn.init.kaiming_normal_(self.x2.weight)
        self.dropout2 = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = self.x1(x)
        x = self.dropout1(x)
        x = self.x2(x)
        x = self.dropout2(x)
        return x


class MTClassifier3D(torch.nn.Module):

    def __init__(self, args):
        super().__init__()
        # Initialize Stem to adjust number of channels
        self.stem = nn.Conv3d(1, 3, (1, 3, 3), stride=1, padding=(0, 1, 1))
        # Initialize 3D Resnet Model
        self.resnet3d = None
        if args.resnet3d_model == "r3d_18":
              # 18 layer Resnet3D
            self.resnet3d = r3d_18(pretrained=True)
        elif args.resnet3d_model == "mc3_18":
            # 18 layer Mixed Convolution network
            self.resnet3d = mc3_18(pretrained=True)
        else:
            # 18 layer deep R(2+1)D network
            self.resnet3d = r2plus1d_18(pretrained=True)

        self.resnet3d_out_features = self.resnet3d.fc.out_features
        
        self.features = args.features

        # # FC layers between resnet3d and the heads
        self.x1 = nn.Linear(self.resnet3d_out_features,
                            self.resnet3d_out_features)
        nn.init.kaiming_normal_(self.x1.weight)
        self.dropout1 = nn.Dropout(p=0.2)
        self.x2 = nn.Linear(self.resnet3d_out_features,
                            self.resnet3d_out_features // 2)
        nn.init.kaiming_normal_(self.x2.weight)
        self.dropout2 = nn.Dropout(p=0.2)

        for feature in self.features:
            setattr(self, f"{feature}_head", ClassifierHead(self.resnet3d_out_features // 2, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.resnet3d(x)
        x = F.relu(self.x1(x))
        x = self.dropout1(x)
        x = F.relu(self.x2(x))
        x = self.dropout2(x)

        head_preds = []
        for feature in self.features:
            head = getattr(self, f"{feature}_head")
            head_pred = head(x)
            head_preds.append(head_pred)

        return head_preds


if __name__ == "__main__":
    # Non exhaustive test for MultiTaskResnet3dClassifier
    parser = MTClassifier3DTrainArgParser()
    args = parser.parse_args()
    model = MTClassifier3D(args).to(args.device)
    model.train()
    dummy_input = torch.zeros((2, 3, 30, 400, 400)).to(args.device)
    dummy_preds = model.forward(dummy_input)
    print(dummy_preds.shape)
