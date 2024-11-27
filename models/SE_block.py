import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # Global Average Pooling
        y = self.fc1(y)  # Excitation
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y  # Scale output
class Bottleneck(nn.Module):
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
        # Add SE Block
        self.se = SEBlock(planes * self.expansion, reduction=reduction)
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
        # Apply SE Block
        out = self.se(out)
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
        # Load ResNet50 weights if pretrained
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.model = resnet50(weights=weights)
        # Set the inplanes attribute to match ResNet's expectations for each layer
        self.model.inplanes = 64  # Initial input channels of ResNet
        # Replace Bottleneck blocks with SE-enabled Bottleneck
        self.model.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.model.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.model.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.model.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.feat_dim = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove fully connected layer
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.embedding_dim)
        )
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.model.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.model.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.model.inplanes, planes, stride, downsample))
        self.model.inplanes = planes * block.expansion  # Update inplanes to match output channels
        for _ in range(1, blocks):
            layers.append(block(self.model.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        print(f"ResNetSE input shape: {x.shape}")
        f = self.model(x)
        f = f.view(-1, self.feat_dim)
        print(f"Feature shape after ResNet model: {f.shape}")
        if self.use_norm:
            f = F.normalize(f, dim=1)
        g = self.projector(f)
        print(f"Output shape after projector: {g.shape}")
        
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g
if __name__ == "__main__":
    # Instantiate the model with specific embedding dimensions
    model = ResNetSE(embedding_dim=128)
    # Test data
    img = torch.randn(4, 3, 224, 224)  # [batch_size, channels, H, W]
    output = model(img)
    print("Final output:", output)




""" 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # Global Average Pooling
        y = self.fc1(y)  # Excitation
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y  # Scale output
class Bottleneck(nn.Module):
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
        # Add SE Block
        self.se = SEBlock(planes * self.expansion, reduction=reduction)
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
        # Apply SE Block
        out = self.se(out)
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
        # Load ResNet50 weights if pretrained
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.model = resnet50(weights=weights)
        # Set the inplanes attribute to match ResNet's expectations for each layer
        self.model.inplanes = 64  # Initial input channels of ResNet
        # Replace Bottleneck blocks with SE-enabled Bottleneck
        self.model.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.model.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.model.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.model.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.feat_dim = self.model.fc.in_features
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove fully connected layer
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.embedding_dim)
        )
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.model.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.model.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.model.inplanes, planes, stride, downsample))
        self.model.inplanes = planes * block.expansion  # Update inplanes to match output channels
        for _ in range(1, blocks):
            layers.append(block(self.model.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        print(f"ResNetSE input shape: {x.shape}")
        f = self.model(x)
        f = f.view(-1, self.feat_dim)
        print(f"Feature shape after ResNet model: {f.shape}")
        if self.use_norm:
            f = F.normalize(f, dim=1)
        g = self.projector(f)
        print(f"Output shape after projector: {g.shape}")
        
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g
if __name__ == "__main__":
    # Instantiate the model with specific embedding dimensions
    model = ResNetSE(embedding_dim=128)
    # Test data
    img = torch.randn(4, 3, 224, 224)  # [batch_size, channels, H, W]
    output = model(img)
    print("Final output:", output)

 """

""" 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2)  # Global Average Pooling
        y = self.fc1(y)  # Excitation
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y  # Scale output
     """