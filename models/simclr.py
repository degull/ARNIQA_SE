import torch
import torch.nn as nn
from dotmap import DotMap
import sys
import os
import torch.nn.functional as F

# 현재 파일의 상위 디렉토리를 PYTHONPATH에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ResNet 대신 ResNetSE를 가져옴
from models.resnet import ResNetSE


class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super(SimCLR, self).__init__()
        # ResNetSE 인코더 사용
        self.encoder = ResNetSE(
            embedding_dim=encoder_params.embedding_dim,
            pretrained=encoder_params.pretrained,
            use_norm=encoder_params.use_norm
        )
        self.temperature = temperature

    def forward(self, img_A, img_B):
        if img_A.dim() != 5 or img_B.dim() != 5:
            raise ValueError("Input images must have dimensions [batch_size, num_crops, 3, H, W].")
        batch_size, num_crops, C, H, W = img_A.size()

        # 디버깅: 변환 전 차원 확인
        print(f"Before view - img_A shape: {img_A.shape}, img_B shape: {img_B.shape}")

        # `contiguous()` 호출 후 `view` 사용
        proj_A = self.encoder(img_A.contiguous().view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]
        proj_B = self.encoder(img_B.contiguous().view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]

        # proj_A와 proj_B가 tuple인 경우 첫 번째 요소 선택
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]
        if isinstance(proj_B, tuple):
            proj_B = proj_B[0]

        # 디버깅: 변환 후 차원 확인
        print(f"After view - proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")
        return proj_A, proj_B

    def compute_loss(self, proj_A, proj_B, hard_negatives):
        # Ensure numerical stability
        proj_A = torch.clamp(proj_A, min=1e-6, max=1 - 1e-6)
        proj_B = torch.clamp(proj_B, min=1e-6, max=1 - 1e-6)

        # NT-Xent Loss 계산
        logits = torch.mm(proj_A, proj_B.T) / self.temperature
        labels = torch.arange(proj_A.size(0)).to(proj_A.device)

        # Numerical stability for softmax
        logits_max = torch.max(logits, dim=1, keepdim=True).values
        logits = logits - logits_max.detach()

        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss

    def nt_xent_loss(self, proj_q, proj_p, hard_negatives):
        tau = 0.5  # Temperature parameter

        print(f"proj_q shape: {proj_q.shape}, hard_negatives shape: {hard_negatives.shape}")

        # Reshape hard_negatives to 4D for ResNetSE
        batch_size, num_crops, C, H, W = hard_negatives.size()
        hard_negatives = hard_negatives.view(-1, C, H, W)  # [batch_size * num_crops, C, H, W]

        # Pass hard_negatives through the encoder
        hard_negatives = self.encoder(hard_negatives)

        if isinstance(hard_negatives, tuple):
            hard_negatives = hard_negatives[0]

        hard_negatives = hard_negatives.view(-1, proj_q.size(1))  # [batch_size * num_crops, embedding_dim]

        # Positive similarity
        positive_sim = torch.exp((proj_q * proj_p).sum(dim=-1) / tau)

        # Negative similarity
        negative_sim = torch.exp((proj_q @ hard_negatives.T) / tau).sum(dim=-1)

        # Clamp values for numerical stability
        epsilon = 1e-8
        positive_sim = torch.clamp(positive_sim, min=epsilon)
        negative_sim = torch.clamp(negative_sim, min=epsilon)

        # NT-Xent loss calculation
        loss = -torch.log(positive_sim / (positive_sim + negative_sim)).mean()

        return loss


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

    # 임시 데이터 생성
    img_A = torch.randn(2, 5, 3, 224, 224)
    img_B = torch.randn(2, 5, 3, 224, 224)

    # 모델의 forward 함수 테스트
    proj_A, proj_B = model(img_A, img_B)
    print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")
