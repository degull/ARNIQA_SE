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
