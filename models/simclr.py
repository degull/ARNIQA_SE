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

import torch
import torch.nn as nn
from dotmap import DotMap
from models.resnet import ResNetSE

class SimCLR(nn.Module):
    def __init__(self, encoder_params: DotMap, temperature: float = 0.1):
        super(SimCLR, self).__init__()
        self.encoder = ResNetSE(embedding_dim=encoder_params.embedding_dim,
                                pretrained=encoder_params.pretrained,
                                use_norm=encoder_params.use_norm)
        self.temperature = temperature

    def forward(self, img_A, img_B):
        if img_A.dim() != 5 or img_B.dim() != 5:
            raise ValueError("Input images must have dimensions [batch_size, num_crops, 3, H, W].")
        batch_size, num_crops, C, H, W = img_A.size()
        proj_A, _ = self.encoder(img_A.view(-1, C, H, W))
        proj_B, _ = self.encoder(img_B.view(-1, C, H, W))
        return proj_A.view(batch_size, num_crops, -1), proj_B.view(batch_size, num_crops, -1)

    def compute_loss(self, proj_q, proj_p):
        return self.nt_xent_loss(proj_q, proj_p)

    def nt_xent_loss(self, a: torch.Tensor, b: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
        # Ensure a and b are normalized
        a = a.view(a.size(0), -1)  # Flatten to [batch_size, embed_dim]
        b = b.view(b.size(0), -1)

        a_norm = a.norm(dim=1, keepdim=True) + 1e-8  # Avoid division by zero
        b_norm = b.norm(dim=1, keepdim=True) + 1e-8

        a_cap = a / a_norm
        b_cap = b / b_norm

        # Combine embeddings
        a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)

        # Compute similarity matrix
        sim = torch.mm(a_cap_b_cap, a_cap_b_cap.t()) / tau

        # Mask to avoid self-similarity in numerator
        batch_size = a_cap.size(0)
        mask = torch.eye(batch_size * 2, device=a.device).bool()

        # Exponentiate and mask diagonal
        exp_sim = torch.exp(sim)
        exp_sim = exp_sim.masked_fill(mask, 0)

        # Calculate loss
        numerators = torch.exp(sim[~mask].view(batch_size * 2, -1))
        denominators = exp_sim.sum(dim=1, keepdim=True)  # 여기서 keepdim=True 추가

        loss = -torch.log(numerators / denominators)
        print(f"Numerators shape: {numerators.shape}, Denominators shape: {denominators.shape}")

        
        return loss.mean()
