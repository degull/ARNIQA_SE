import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # Global Average Pooling
        y = self.fc1(y)  # FC Layer 1
        y = self.relu(y)
        y = self.fc2(y)  # FC Layer 2
        y = self.sigmoid(y).view(b, c, 1, 1)  # Reshape for scaling
        return x * y  # Scale the input features


class Bottleneck(nn.Module):
    """
    Bottleneck block with Squeeze-and-Excitation (SE) block
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.se = SEBlock(planes * self.expansion, reduction=reduction)  # Add SE Block

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)  # Apply SE block

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetSE(nn.Module):
    def __init__(self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True):
        super(ResNetSE, self).__init__()
        self.pretrained = pretrained
        self.use_norm = use_norm
        self.embedding_dim = embedding_dim

        # Load ResNet-50 with pretrained weights if specified
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.model = resnet50(weights=weights)

        # Set initial inplanes value for Bottleneck layers
        self.inplanes = 64

        # Replace Bottleneck blocks with SE-enabled Bottleneck blocks
        self.model.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.model.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.model.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.model.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # Get feature dimension for the final layer
        self.feat_dim = self.model.fc.in_features

        # Remove the original fully connected layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        # Define a projection head for embeddings
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.embedding_dim)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Create a layer of Bottleneck blocks with SE blocks
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion  # Update inplanes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for ResNetSE
        """
        f = self.model(x)  # Extract features
        f = f.view(f.size(0), -1)  # Flatten features
        if self.use_norm:
            f = F.normalize(f, dim=1)
        g = self.projector(f)
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g
