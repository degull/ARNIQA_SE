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
import torch
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
