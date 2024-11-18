import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Squeeze-and-Excitation Block
        :param channels: 입력 채널 수
        :param reduction: 채널 축소 비율
        """
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()  # (batch_size, channels, height, width)
        y = x.view(b, c, -1).mean(dim=2)  # Global Average Pooling (Squeeze 단계)
        y = self.fc1(y)  # 첫 번째 FC 레이어
        y = self.relu(y)  # ReLU 활성화 함수
        y = self.fc2(y)  # 두 번째 FC 레이어
        y = self.sigmoid(y).view(b, c, 1, 1)  # Excitation 단계
        return x * y  # 중요도 벡터 y를 원래 입력 텐서 x에 곱함
