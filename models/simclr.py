import torch
import torch.nn as nn
from dotmap import DotMap
import sys
import os

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

        # Encoder를 통해 feature 추출
        proj_A = self.encoder(img_A.view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]
        proj_B = self.encoder(img_B.view(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]

        # proj_A와 proj_B가 tuple인 경우 첫 번째 요소 선택
        if isinstance(proj_A, tuple):
            proj_A = proj_A[0]
        if isinstance(proj_B, tuple):
            proj_B = proj_B[0]

        # 디버깅: 변환 후 차원 확인
        print(f"After view - proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")
        return proj_A, proj_B

    def compute_loss(self, proj_q, proj_p):
        return self.nt_xent_loss(proj_q, proj_p)

    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        a_norm = torch.norm(a, dim=1).reshape(-1, 1)
        b_norm = torch.norm(b, dim=1).reshape(-1, 1)
        a_cap = torch.div(a, a_norm)
        b_cap = torch.div(b, b_norm)

        sim = torch.mm(a_cap, b_cap.t()) / tau
        exp_sim = torch.exp(sim)

        # Debugging
        print(f"[DEBUG] Cosine Similarity mean/std: {sim.mean().item()} / {sim.std().item()}")
        print(f"[DEBUG] exp_sim mean/std: {exp_sim.mean().item()} / {exp_sim.std().item()}")

        pos_sim = torch.exp(torch.sum(a_cap * b_cap, dim=1) / tau)
        denominator = exp_sim.sum(dim=1) - pos_sim

        # Debugging
        print(f"[DEBUG] pos_sim mean/std: {pos_sim.mean().item()} / {pos_sim.std().item()}")
        print(f"[DEBUG] denominator mean/std: {denominator.mean().item()} / {denominator.std().item()}")

        loss = -torch.log(pos_sim / denominator).mean()
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
