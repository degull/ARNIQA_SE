import torch.nn as nn
import torch.nn.functional as F
import timm  # Timm 라이브러리 사용
import torch

class EfficientFormerModel(nn.Module):
    """
    EfficientFormer model with a projection head.

    Args:
        embedding_dim (int): embedding dimension of the projection head
        pretrained (bool): whether to use pretrained weights
        use_norm (bool): whether to normalize the embeddings
    """
    def __init__(self, embedding_dim: int, pretrained: bool = True, use_norm: bool = True):
        super(EfficientFormerModel, self).__init__()

        self.pretrained = pretrained
        self.use_norm = use_norm
        self.embedding_dim = embedding_dim

        # EfficientFormer-L1을 timm 라이브러리에서 불러오기 (사전학습 가중치 사용)
        if self.pretrained:
            self.model = timm.create_model('efficientformer_l1', pretrained=True, num_classes=0)  # num_classes=0으로 마지막 레이어 제거
        else:
            self.model = timm.create_model('efficientformer_l1', pretrained=False, num_classes=0)

        # Feature dimension (모델 마지막 출력 차원)
        self.feat_dim = self.model.num_features

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.embedding_dim)
        )

    def forward(self, x):
        # EfficientFormer로 특징 추출
        f = self.model(x)  # EfficientFormer의 출력

        if self.use_norm:
            f = F.normalize(f, dim=1)  # 특징 벡터 정규화

        # Projection head를 통해 임베딩 생성
        g = self.projector(f)
        if self.use_norm:
            return f, F.normalize(g, dim=1)  # 정규화된 임베딩 반환
        else:
            return f, g  # 정규화하지 않은 임베딩 반환
