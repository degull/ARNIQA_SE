""" # simclr.py
import torch
import torch.nn as nn
from dotmap import DotMap
import sys
import os
from models.resnet import ResNetSE

# 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super(SimCLR, self).__init__()
        # ResNetSE를 encoder로 초기화
        self.encoder = ResNetSE(embedding_dim=encoder_params.embedding_dim,
                                pretrained=encoder_params.pretrained,
                                use_norm=encoder_params.use_norm)
        self.temperature = temperature

    def forward(self, img_A, img_B):
        if img_A.dim() != 5 or img_B.dim() != 5:
            raise ValueError("Input images must have dimensions [batch_size, num_crops, 3, H, W].")

        batch_size, num_crops, C, H, W = img_A.size()

        proj_A = self.encoder(img_A.reshape(-1, C, H, W))
        proj_B = self.encoder(img_B.reshape(-1, C, H, W))

        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]
        if isinstance(proj_B, tuple):
            proj_B = proj_B[0]

        return proj_A, proj_B

    def compute_loss(self, proj_q, proj_p):
        return self.nt_xent_loss(proj_q, proj_p)

    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        a_norm = torch.norm(a, dim=1).reshape(-1, 1)
        a_cap = torch.div(a, a_norm)
        b_norm = torch.norm(b, dim=1).reshape(-1, 1)
        b_cap = torch.div(b, b_norm)

        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
        sim = torch.mm(a_cap_b_cap, a_cap_b_cap.transpose(0, 1)) / tau
        exp_sim_by_tau = torch.exp(sim)

        sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
        exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
        numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, a_cap_b_cap), tau))
        denominators = sum_of_rows - exp_sim_by_tau_diag
        neglog_num_by_den = -torch.log(numerators / denominators)
        
        return torch.mean(neglog_num_by_den)


if __name__ == "__main__":
    import torch
    from dotmap import DotMap

    encoder_params = DotMap({
        'embedding_dim': 128,
        'pretrained': False,
        'use_norm': True
    })

    model = SimCLR(encoder_params)

    img_A = torch.randn(2, 5, 3, 224, 224)
    img_B = torch.randn(2, 5, 3, 224, 224)

    proj_A, proj_B = model(img_A, img_B)
    print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")
 """

# SimCLR 수정
""" import torch
import torch.nn as nn
from dotmap import DotMap
from models.resnet import ResNetSE


class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.5):
        super(SimCLR, self).__init__()
        self.encoder = ResNetSE(
            embedding_dim=encoder_params.embedding_dim,
            pretrained=encoder_params.pretrained,
            use_norm=encoder_params.use_norm
        )
        self.temperature = temperature

    def forward(self, img_A, img_B):
        if img_A.dim() != 5 or img_B.dim() != 5:
            raise ValueError(
                f"Input images must have dimensions [batch_size, num_crops, 3, H, W], "
                f"but got {img_A.size()} and {img_B.size()}."
            )

        batch_size, num_crops, C, H, W = img_A.size()

        proj_A, _ = self.encoder(img_A.view(-1, C, H, W))
        proj_B, _ = self.encoder(img_B.view(-1, C, H, W))

        return proj_A.view(batch_size, num_crops, -1), proj_B.view(batch_size, num_crops, -1)

    def compute_loss(self, proj_q, proj_p):
        return self.nt_xent_loss(proj_q, proj_p)

    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
        a = a.view(a.size(0), -1)  # [batch_size, embedding_dim]
        b = b.view(b.size(0), -1)  # [batch_size, embedding_dim]

        a_norm = a.norm(dim=1, keepdim=True) + 1e-8
        b_norm = b.norm(dim=1, keepdim=True) + 1e-8

        a_cap = a / a_norm  # Normalize a
        b_cap = b / b_norm  # Normalize b

        # Concatenate a and b for similarity calculation
        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)  # [2 * batch_size, embedding_dim]

        sim = torch.mm(a_cap_b_cap, a_cap_b_cap.t()) / tau  # Cosine similarity matrix
        batch_size = a.size(0)

        # Create mask to exclude self-similarities
        mask = torch.eye(2 * batch_size, device=a.device).bool()

        # Correct similarity calculation
        exp_sim = torch.exp(sim)
        exp_sim = exp_sim.masked_fill(mask, 0)  # Set diagonal to 0

        numerators = torch.exp(torch.diag(sim, batch_size) + torch.diag(sim, -batch_size))
        denominators = exp_sim.sum(dim=1, keepdim=True)

        # Adjust size if numerators or denominators mismatch
        if numerators.size(0) != denominators.size(0):
            numerators = numerators[:denominators.size(0)]  # Trim numerators to match

        loss = -torch.log(numerators / denominators)
        return loss.mean()

    def get_gradcam_target_layer(self):
        # Return the target layer for Grad-CAM visualization (e.g., last convolutional layer)
        return self.encoder.layer4[2].conv3
 """


## 지울거
""" import torch
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

""" import torch
import torch.nn as nn
from dotmap import DotMap
import sys
import os
# 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.resnet import ResNet
class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super(SimCLR, self).__init__()
        self.encoder = ResNet(embedding_dim=encoder_params.embedding_dim,
                              pretrained=encoder_params.pretrained,
                              use_norm=encoder_params.use_norm)
        self.temperature = temperature
    def forward(self, img_A, img_B):
        # img_A와 img_B의 차원이 [batch_size, num_crops, C, H, W]일 것으로 가정
        if img_A.dim() != 5 or img_B.dim() != 5:
            raise ValueError("Input images must have dimensions [batch_size, num_crops, 3, H, W].")
        batch_size, num_crops, C, H, W = img_A.size()  # [batch_size, num_crops, 3, 224, 224]
        # 디버깅: 변환 전 차원 확인
        print(f"Before view - img_A shape: {img_A.shape}, img_B shape: {img_B.shape}")
        # Encoder를 통해 feature 추출
        proj_A = self.encoder(img_A.view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]
        proj_B = self.encoder(img_B.view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]
        # proj_A와 proj_B가 tuple인 경우 첫 번째 요소 선택
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]  # 필요한 경우 첫 번째 요소를 선택
        if isinstance(proj_B, tuple):
            proj_B = proj_B[0]  # 필요한 경우 첫 번째 요소를 선택
        # 디버깅: 변환 후 차원 확인
        print(f"After view - proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")
        return proj_A, proj_B
    def compute_loss(self, proj_q, proj_p):
        # NT-Xent 손실 계산
        return self.nt_xent_loss(proj_q, proj_p)
    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        a_norm = torch.norm(a, dim=1).reshape(-1, 1)
        a_cap = torch.div(a, a_norm)
        b_norm = torch.norm(b, dim=1).reshape(-1, 1)
        b_cap = torch.div(b, b_norm)
        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
        a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
        b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
        sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
        sim_by_tau = torch.div(sim, tau)
        exp_sim_by_tau = torch.exp(sim_by_tau)
        sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
        exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
        numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
        denominators = sum_of_rows - exp_sim_by_tau_diag
        num_by_den = torch.div(numerators, denominators)
        neglog_num_by_den = -torch.log(num_by_den)
        return torch.mean(neglog_num_by_den)
if __name__ == "__main__":
    # 간단한 테스트 케이스
    import torch
    from dotmap import DotMap
    # 임시 encoder 파라미터 설정
    encoder_params = DotMap({
        'embedding_dim': 128,
        'pretrained': False,
        'use_norm': True
    })
    # SimCLR 모델 초기화
    model = SimCLR(encoder_params)
    # 임시 데이터 생성: [batch_size, num_crops, C, H, W] 크기의 텐서
    img_A = torch.randn(2, 5, 3, 224, 224)
    img_B = torch.randn(2, 5, 3, 224, 224)
    # 모델의 forward 함수 테스트
    proj_A, proj_B = model(img_A, img_B)
    # 결과 출력
    print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")
 """

## 지울거
import torch
import torch.nn as nn
from dotmap import DotMap
import sys
import os
# 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.resnet import ResNet
class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super(SimCLR, self).__init__()
        self.encoder = ResNet(embedding_dim=encoder_params.embedding_dim,
                              pretrained=encoder_params.pretrained,
                              use_norm=encoder_params.use_norm)
        self.temperature = temperature
    def forward(self, img_A, img_B):
        # img_A와 img_B의 차원이 [batch_size, num_crops, C, H, W]일 것으로 가정
        if img_A.dim() != 5 or img_B.dim() != 5:
            raise ValueError("Input images must have dimensions [batch_size, num_crops, 3, H, W].")
        batch_size, num_crops, C, H, W = img_A.size()  # [batch_size, num_crops, 3, 224, 224]
        # 디버깅: 변환 전 차원 확인
        print(f"Before view - img_A shape: {img_A.shape}, img_B shape: {img_B.shape}")
        # Encoder를 통해 feature 추출
        proj_A = self.encoder(img_A.view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]
        proj_B = self.encoder(img_B.view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]
        # proj_A와 proj_B가 tuple인 경우 첫 번째 요소 선택
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]  # 필요한 경우 첫 번째 요소를 선택
        if isinstance(proj_B, tuple):
            proj_B = proj_B[0]  # 필요한 경우 첫 번째 요소를 선택
        # 디버깅: 변환 후 차원 확인
        print(f"After view - proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")
        return proj_A, proj_B
    def compute_loss(self, proj_q, proj_p):
        # NT-Xent 손실 계산
        return self.nt_xent_loss(proj_q, proj_p)
    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        a_norm = torch.norm(a, dim=1).reshape(-1, 1)
        a_cap = torch.div(a, a_norm)
        b_norm = torch.norm(b, dim=1).reshape(-1, 1)
        b_cap = torch.div(b, b_norm)
        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
        a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
        b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
        sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
        sim_by_tau = torch.div(sim, tau)
        exp_sim_by_tau = torch.exp(sim_by_tau)
        sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
        exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
        numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
        denominators = sum_of_rows - exp_sim_by_tau_diag
        num_by_den = torch.div(numerators, denominators)
        neglog_num_by_den = -torch.log(num_by_den)
        return torch.mean(neglog_num_by_den)
if __name__ == "__main__":
    # 간단한 테스트 케이스
    import torch
    from dotmap import DotMap
    # 임시 encoder 파라미터 설정
    encoder_params = DotMap({
        'embedding_dim': 128,
        'pretrained': False,
        'use_norm': True
    })
    # SimCLR 모델 초기화
    model = SimCLR(encoder_params)
    # 임시 데이터 생성: [batch_size, num_crops, C, H, W] 크기의 텐서
    img_A = torch.randn(2, 5, 3, 224, 224)
    img_B = torch.randn(2, 5, 3, 224, 224)
    # 모델의 forward 함수 테스트
    proj_A, proj_B = model(img_A, img_B)
    # 결과 출력
    print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")