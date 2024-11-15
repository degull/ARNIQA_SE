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
        proj_A = self.encoder(img_A.reshape(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]
        proj_B = self.encoder(img_B.reshape(-1, C, H, W))  # [batch_size * num_crops, embedding_dim]

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

    #def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    #    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    #    a_cap = torch.div(a, a_norm)
    #    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    #    b_cap = torch.div(b, b_norm)
    #    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    #    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    #    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    #    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    #    sim_by_tau = torch.div(sim, tau)
    #    exp_sim_by_tau = torch.exp(sim_by_tau)
    #    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    #    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    #    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), tau))
    #    denominators = sum_of_rows - exp_sim_by_tau_diag
    #    num_by_den = torch.div(numerators, denominators)
    #    neglog_num_by_den = -torch.log(num_by_den)
    #    return torch.mean(neglog_num_by_den)
    


    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        # 벡터 정규화
        a_norm = torch.norm(a, dim=1).reshape(-1, 1)
        a_cap = torch.div(a, a_norm)
        b_norm = torch.norm(b, dim=1).reshape(-1, 1)
        b_cap = torch.div(b, b_norm)
        
        # 양성/음성 쌍 유사도 계산
        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
        sim = torch.mm(a_cap_b_cap, a_cap_b_cap.transpose(0, 1)) / tau
        exp_sim_by_tau = torch.exp(sim)
        
        # 대조 손실 계산
        sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
        exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
        numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, a_cap_b_cap), tau))
        denominators = sum_of_rows - exp_sim_by_tau_diag
        neglog_num_by_den = -torch.log(numerators / denominators)
        
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