""" # resnet.py
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
from models.SE_block import SEBlock  # SE_block.py에서 SEBlock 가져오기

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

        # Load ResNet50 with SE Bottleneck blocks
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.model = resnet50(weights=weights)
        
        # Set inplanes for ResNet compatibility
        self.model.inplanes = 64

        # Replace Bottleneck blocks with SE-enabled Bottleneck
        self.model.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.model.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.model.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.model.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # Define feature and projection layers
        self.feat_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove fully connected layer
        
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
        f = self.model(x)
        f = f.view(-1, self.feat_dim)

        if self.use_norm:
            f = nn.functional.normalize(f, dim=1)

        g = self.projector(f)
        
        if self.use_norm:
            return f, nn.functional.normalize(g, dim=1)
        else:
            return f, g
 """

##이거원본
""" import os
import sys
sys.path.append(os.path.abspath("E:/ARNIQA - SE/ARNIQA/models"))  # 절대경로를 사용하여 경로 추가

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50
from models.SE_block import SEBlock  # SEBlock import

class Bottleneck(nn.Module):
    expansion = 4  # Bottleneck block의 확장 계수

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

        # SEBlock의 입력 채널 수는 conv3의 출력과 동일
        self.se = SEBlock(planes * self.expansion, reduction=reduction)

        self.stride = stride

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

        # SEBlock을 적용
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    
class ResNetSE(nn.Module):
    def __init__(self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True):
        super(ResNetSE, self).__init__()
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained
        self.use_norm = use_norm

        # ResNet-50의 weights를 가져옵니다
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.model = resnet50(weights=weights)

        # Bottleneck 블록에서 inplanes 초기값
        self.model.inplanes = 64

        # Bottleneck 블록을 SEBlock이 포함된 버전으로 대체
        self.model.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.model.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.model.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.model.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # 마지막 FC layer 제거
        self.feat_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()

        # Projector layer 정의
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.embedding_dim)
        )

        self.init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.model.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.model.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.model.inplanes, planes, stride, downsample)]
        self.model.inplanes = planes * block.expansion
        layers.extend([block(self.model.inplanes, planes) for _ in range(1, blocks)])

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        f = self.model(x)
        f = f.view(-1, self.feat_dim)

        if self.use_norm:
            f = nn.functional.normalize(f, dim=1)

        g = self.projector(f)
        return f, nn.functional.normalize(g, dim=1) if self.use_norm else g

# ResNetSE 모델 구조를 출력하는 코드 추가
if __name__ == "__main__":
    # 모델 초기화
    model = ResNetSE(embedding_dim=128, pretrained=True)
    
    # 모델 구조 출력
    print(model)
 """


### 지울거
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50
import torch

class ResNet(nn.Module):
    """
    ResNet model with a projection head.
    Args:
        embedding_dim (int): embedding dimension of the projection head
        pretrained (bool): whether to use pretrained weights
        use_norm (bool): whether to normalize the embeddings
    """
    def __init__(self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.use_norm = use_norm
        self.embedding_dim = embedding_dim

        # Load ResNet-50 with or without pretrained weights
        if self.pretrained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.model = resnet50(weights=weights)

        # Extract feature dimension from ResNet
        self.feat_dim = self.model.fc.in_features

        # Remove the fully connected layer and replace with a custom projection head
        self.model = nn.Sequential(*list(self.model.children())[:-1])  # Remove FC layer and AvgPool

        # Define the projection head
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim * 2),  # 차원 확장
            nn.ReLU(),
            nn.Linear(self.feat_dim * 2, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),  # 배치 정규화 추가
        )


    def forward(self, x):
        # Extract features from ResNet
        f = self.model(x)  # Feature map from ResNet
        f = f.view(-1, self.feat_dim)  # Flatten to [batch_size, feat_dim]

        # Normalize if use_norm is True
        if self.use_norm:
            f = F.normalize(f, dim=1)

        # Pass through the projection head
        g = self.projector(f)

        # Normalize the output of the projection head if use_norm is True
        if self.use_norm:
            return f, F.normalize(g, dim=1)
        else:
            return f, g

# Testing the model
if __name__ == "__main__":
    # Create ResNet instance with normalization enabled/disabled
    model_with_norm = ResNet(embedding_dim=128, pretrained=True, use_norm=True)
    model_without_norm = ResNet(embedding_dim=128, pretrained=True, use_norm=False)

    # Sample input
    sample_input = torch.randn(1, 3, 224, 224)

    # Test forward pass with normalization enabled
    f_norm, g_norm = model_with_norm(sample_input)
    print(f"With normalization - f shape: {f_norm.shape}, g shape: {g_norm.shape}")

    # Test forward pass with normalization disabled
    f_no_norm, g_no_norm = model_without_norm(sample_input)
    print(f"Without normalization - f shape: {f_no_norm.shape}, g shape: {g_no_norm.shape}")
