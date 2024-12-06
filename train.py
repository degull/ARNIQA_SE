## proj_A와 proj_B :  양성 쌍으로 가까워져야 함
## proj_A와 proj_B : 평균은 줄어들어야 함
## 양성 쌍 훈련에서는 proj_A와 proj_B가 서로 같은 이미지를 나타내니까, 모델은 두 벡터의 거리가 가까워지도록 학습해야 함

# kadid ver1 (ridge regressor)
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
from torch.cuda.amp import custom_fwd
import matplotlib.pyplot as plt


def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)


def calculate_srcc_plcc(proj_A, proj_B):

    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()

    # 디버깅: 입력 크기 확인
    print(f"calculate_srcc_plcc: proj_A shape {proj_A.shape}, proj_B shape {proj_B.shape}")

    # 데이터 타입 변경으로 메모리 사용량 감소
    proj_A = proj_A.astype(np.float32)
    proj_B = proj_B.astype(np.float32)

    # 데이터 크기 줄이기 위해 차원 축소 (예: PCA)
    from sklearn.decomposition import PCA

    n_samples = proj_A.shape[0]
    n_features = proj_A.shape[1]
    max_components = min(n_samples, n_features)

    n_components = min(128, max_components)

    if n_components < max_components:
        pca = PCA(n_components=n_components)
        proj_A = pca.fit_transform(proj_A)
        proj_B = pca.transform(proj_B)
        print(f"After PCA: proj_A shape {proj_A.shape}, proj_B shape {proj_B.shape}")

    # Spearman 및 Pearson 계산
    srocc_list = []
    plcc_list = []
    for i in range(proj_A.shape[0]):  # 배치 단위로 계산
        srocc, _ = stats.spearmanr(proj_A[i], proj_B[i])
        plcc, _ = stats.pearsonr(proj_A[i], proj_B[i])
        srocc_list.append(srocc)
        plcc_list.append(plcc)

    # 평균 값 반환
    avg_srocc = np.mean(srocc_list)
    avg_plcc = np.mean(plcc_list)
    return avg_srocc, avg_plcc




def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # 입력 데이터가 5차원([batch_size, C, H, W])인 경우에만 unsqueeze 호출
            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)  # Add num_crops dimension
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)  # Add num_crops dimension

            # num_crops 차원을 2로 확장
            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            proj_A, proj_B = model(inputs_A, inputs_B)
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)

            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc



def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.StepLR,
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            inputs_A = apply_random_distortions(inputs_A)
            inputs_B = apply_random_distortions(inputs_B)

            # 입력 데이터가 5차원([batch_size, C, H, W])인 경우에만 unsqueeze 호출
            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)

            # num_crops 차원을 2로 확장
            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B, hard_negatives)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)


        lr_scheduler.step()

        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}] Validation Results: SRCC = {avg_srocc_val:.4f}, PLCC = {avg_plcc_val:.4f}")

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

    print("Finished training")

def train_ridge_regressor(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_dataloader):
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            # Ensure inputs_A has the correct dimensions
            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)

            proj_A, _ = model(inputs_A, inputs_A)

            # 디버깅: proj_A와 mos 크기 출력
            print(f"[Batch {batch_idx}] proj_A shape: {proj_A.shape}, mos shape: {mos.shape}")

            embeddings.append(proj_A.cpu().numpy())
            
            # mos를 proj_A와 같은 길이로 확장
            mos_repeated = np.repeat(mos.numpy(), proj_A.shape[0] // mos.shape[0])
            mos_scores.append(mos_repeated)

    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)

    # 디버깅: 최종 크기 출력
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Final MOS scores shape: {mos_scores.shape}")

    regressor = Ridge(alpha=1.0)
    regressor.fit(embeddings, mos_scores)
    return regressor




def evaluate_ridge_regressor(regressor, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores, predictions = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            # Ensure inputs_A has the correct dimensions
            if inputs_A.dim() == 4:  # [batch_size, C, H, W]
                inputs_A = inputs_A.unsqueeze(1)  # Add num_crops dimension -> [batch_size, 1, C, H, W]
                inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)  # Expand num_crops to 2

            proj_A, _ = model(inputs_A, inputs_A)

            embeddings.append(proj_A.cpu().numpy())
            mos_scores.append(mos.numpy())

            preds = regressor.predict(proj_A.cpu().numpy())
            predictions.append(preds)

    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)
    predictions = np.hstack(predictions)

    # 디버깅: 배열 크기 확인
    print(f"Final MOS scores shape: {mos_scores.shape}")
    print(f"Final Predictions shape: {predictions.shape}")

    return mos_scores, predictions


def plot_results(mos_scores, predictions):
    # 디버깅: 입력 크기 출력
    print(f"Plotting: MOS scores shape: {mos_scores.shape}, Predictions shape: {predictions.shape}")

    # 크기 동기화
    min_length = min(len(mos_scores), len(predictions))
    mos_scores = mos_scores[:min_length]
    predictions = predictions[:min_length]

    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Ridge Regressor Performance')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    args = DotMap({
        "data_base_path": "E:/ARNIQA/dataset",
        "training": {
            "epochs": 10,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "num_workers": 4,
        },
        "checkpoint_path": "E:/ARNIQA/experiments/my_experiment/pretrain",
        "model": {
            "temperature": 0.1,
            "embedding_dim": 128
        }
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KADID10KDataset("E:/ARNIQA - SE/ARNIQA/dataset/KADID10K/kadid10k.csv")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size * 2,
        shuffle=True,
        num_workers=args.training.num_workers,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.training.num_workers)

    model = SimCLR(encoder_params=DotMap({"embedding_dim": args.model.embedding_dim, "pretrained": True}), temperature=args.model.temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    print("Training Ridge Regressor...")
    regressor = train_ridge_regressor(model, train_dataloader, device)

    print("Evaluating Ridge Regressor...")
    mos_scores, predictions = evaluate_ridge_regressor(regressor, model, val_dataloader, device)

    plot_results(mos_scores, predictions)

 """

# # kadid ver1 (ridge regressor + 검증) -? srcc&plcc 완 (but 그래프 오류)
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
from sklearn.linear_model import Ridge
from scipy import stats
import yaml

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def verify_positive_pairs(distortions_A, distortions_B):
    
    #양성쌍이 동일한 왜곡이 적용되었는지 확인합니다.
    
    if distortions_A == distortions_B:
        print("[Positive Pair Verification] Success: Distortions match.")
    else:
        print("[Positive Pair Verification] Error: Distortions do not match.")
        print(f"distortions_A: {distortions_A}, distortions_B: {distortions_B}")

def verify_hard_negatives(original_shape, downscaled_shape):
    
    #경음성쌍이 정확히 50%로 다운스케일 되었는지 확인합니다.
    
    expected_shape = (original_shape[-2] // 2, original_shape[-1] // 2)
    if downscaled_shape[-2:] == expected_shape:
        print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
    else:
        print("[Hard Negative Verification] Error: Hard negatives are not correctly downscaled.")
        print(f"Expected: {expected_shape}, Got: {downscaled_shape[-2:]}")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    
    # Ensure shapes are consistent
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"

    # Flatten both vectors for overall SRCC/PLCC
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    
    return srocc, plcc



def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)

            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            proj_A, proj_B = model(inputs_A, inputs_B)
            srocc, _ = stats.spearmanr(proj_A.flatten().cpu(), proj_B.flatten().cpu())
            plcc, _ = stats.pearsonr(proj_A.flatten().cpu(), proj_B.flatten().cpu())

            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.StepLR,
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:
    # 수정된 checkpoint_path
    checkpoint_path = Path(str(args.checkpoint_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # 동일한 왜곡을 적용하도록 shared_distortion 설정
            shared_distortion = random.choice(["blur", "noise", "color_shift", "jpeg_compression"])
            inputs_A, distortions_A = apply_random_distortions(inputs_A, shared_distortion=shared_distortion, return_info=True)
            inputs_B, distortions_B = apply_random_distortions(inputs_B, shared_distortion=shared_distortion, return_info=True)

            # 양성 쌍 검증
            if distortions_A == distortions_B:
                print("[Positive Pair Verification] Success: Distortions match.")
            else:
                print(f"[Positive Pair Verification] Error: Distortions do not match.\n"
                      f"distortions_A: {distortions_A}, distortions_B: {distortions_B}")

            # 입력 데이터가 5차원([batch_size, C, H, W])인 경우에만 unsqueeze 호출
            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)

            # num_crops 차원을 2로 확장
            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            # 경음성 쌍 생성 및 검증
            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)
            if inputs_B.shape[0] == hard_negatives.shape[0]:
                print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
            else:
                print(f"[Hard Negative Verification] Error: Shape mismatch.\n"
                      f"inputs_B shape: {inputs_B.shape}, hard_negatives shape: {hard_negatives.shape}")

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B, hard_negatives)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        lr_scheduler.step()
        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}] Validation Results: SRCC = {avg_srocc_val:.4f}, PLCC = {avg_plcc_val:.4f}")

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

    print("Finished training")



def train_ridge_regressor(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_dataloader):
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            # Ensure inputs_A has the correct dimensions
            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)

            proj_A, _ = model(inputs_A, inputs_A)

            # Debug: Check shapes of proj_A and mos
            print(f"[Batch {batch_idx}] proj_A shape: {proj_A.shape}, mos shape: {mos.shape}")

            embeddings.append(proj_A.cpu().numpy())

            # Match mos length with proj_A length
            repeat_factor = proj_A.shape[0] // mos.shape[0]
            mos_repeated = np.repeat(mos.numpy(), repeat_factor)
            if len(mos_repeated) > proj_A.shape[0]:
                mos_repeated = mos_repeated[:proj_A.shape[0]]  # Trim excess
            mos_scores.append(mos_repeated)

    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)

    # Debug: Check final shapes
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Final MOS scores shape: {mos_scores.shape}")

    regressor = Ridge(alpha=1.0)
    regressor.fit(embeddings, mos_scores)
    return regressor


def evaluate_ridge_regressor(regressor, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
                inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)

            proj_A, _ = model(inputs_A, inputs_A)
            predictions.append(regressor.predict(proj_A.cpu().numpy()))
            mos_scores.append(mos.numpy())

    mos_scores = np.hstack(mos_scores)
    predictions = np.hstack(predictions)
    return mos_scores, predictions

def plot_results(mos_scores, predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Ridge Regressor Performance')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":

    # Config 경로 설정
    config_path = "E:/ARNIQA - SE/ARNIQA/config.yaml"
    config = load_config(config_path)

    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(config.data_base_path) / "kadid10k.csv"
    dataset = KADID10KDataset(dataset_path)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=min(config.training.num_workers, 16))
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=min(config.training.num_workers, 16))

    model = SimCLR(encoder_params=DotMap(config.model.encoder), temperature=config.model.temperature).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=config.training.optimizer.momentum, weight_decay=config.training.optimizer.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.training.lr_scheduler.T_0, T_mult=config.training.lr_scheduler.T_mult, eta_min=config.training.lr_scheduler.eta_min)
    scaler = torch.amp.GradScaler()

    train(config, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    regressor = train_ridge_regressor(model, train_dataloader, device)
    mos_scores, predictions = evaluate_ridge_regressor(regressor, model, val_dataloader, device)
    plot_results(mos_scores, predictions)

 """

# # kadid ver2 (ridge regressor + 검증) -> srcc&plcc 완 ( 그래프 생성 완  / but, 수정필요)
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
from sklearn.linear_model import Ridge
from scipy import stats
import yaml

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def verify_positive_pairs(distortions_A, distortions_B):
    
    # 양성쌍이 동일한 왜곡이 적용되었는지 확인합니다.
    
    if distortions_A == distortions_B:
        print("[Positive Pair Verification] Success: Distortions match.")
    else:
        print("[Positive Pair Verification] Error: Distortions do not match.")
        print(f"distortions_A: {distortions_A}, distortions_B: {distortions_B}")

def verify_hard_negatives(original_shape, downscaled_shape):
    
    # 경음성쌍이 정확히 50%로 다운스케일 되었는지 확인합니다.
    
    expected_shape = (original_shape[-2] // 2, original_shape[-1] // 2)
    if downscaled_shape[-2:] == expected_shape:
        print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
    else:
        print("[Hard Negative Verification] Error: Hard negatives are not correctly downscaled.")
        print(f"Expected: {expected_shape}, Got: {downscaled_shape[-2:]}")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    
    # Ensure shapes are consistent
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"

    # Flatten both vectors for overall SRCC/PLCC
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    
    return srocc, plcc



def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)

            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            proj_A, proj_B = model(inputs_A, inputs_B)
            srocc, _ = stats.spearmanr(proj_A.flatten().cpu(), proj_B.flatten().cpu())
            plcc, _ = stats.pearsonr(proj_A.flatten().cpu(), proj_B.flatten().cpu())

            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.StepLR,
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:
    # 수정된 checkpoint_path
    checkpoint_path = Path(str(args.checkpoint_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # 동일한 왜곡을 적용하도록 shared_distortion 설정
            shared_distortion = random.choice(["blur", "noise", "color_shift", "jpeg_compression"])
            inputs_A, distortions_A = apply_random_distortions(inputs_A, shared_distortion=shared_distortion, return_info=True)
            inputs_B, distortions_B = apply_random_distortions(inputs_B, shared_distortion=shared_distortion, return_info=True)

            # 양성 쌍 검증
            if distortions_A == distortions_B:
                print("[Positive Pair Verification] Success: Distortions match.")
            else:
                print(f"[Positive Pair Verification] Error: Distortions do not match.\n"
                      f"distortions_A: {distortions_A}, distortions_B: {distortions_B}")

            # 입력 데이터가 5차원([batch_size, C, H, W])인 경우에만 unsqueeze 호출
            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)

            # num_crops 차원을 2로 확장
            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            # 경음성 쌍 생성 및 검증
            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)
            if inputs_B.shape[0] == hard_negatives.shape[0]:
                print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
            else:
                print(f"[Hard Negative Verification] Error: Shape mismatch.\n"
                      f"inputs_B shape: {inputs_B.shape}, hard_negatives shape: {hard_negatives.shape}")

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B, hard_negatives)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        lr_scheduler.step()
        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}] Validation Results: SRCC = {avg_srocc_val:.4f}, PLCC = {avg_plcc_val:.4f}")

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

    print("Finished training")



def train_ridge_regressor(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_dataloader):
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)

            proj_A, _ = model(inputs_A, inputs_A)

            # Debug: 확인용
            print(f"[Batch {batch_idx}] proj_A shape: {proj_A.shape}, mos shape: {mos.shape}")

            embeddings.append(proj_A.cpu().numpy())

            # MOS 반복 방식 수정
            mos_repeated = np.tile(mos.numpy(), (proj_A.shape[0] // mos.shape[0]))
            mos_repeated = mos_repeated[:proj_A.shape[0]]  # Trim to match proj_A size
            mos_scores.append(mos_repeated)

    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)

    # Debug: 최종 크기 출력
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Final MOS scores shape: {mos_scores.shape}")

    regressor = Ridge(alpha=1.0)
    regressor.fit(embeddings, mos_scores)
    return regressor



def evaluate_ridge_regressor(regressor, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
                inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)

            proj_A, _ = model(inputs_A, inputs_A)
            prediction = regressor.predict(proj_A.cpu().numpy())

            # Debug: 확인용
            print(f"[Batch {batch_idx}] proj_A shape: {proj_A.shape}, prediction shape: {prediction.shape}, mos shape: {mos.shape}")

            # MOS 및 예측값 크기 조정
            mos_repeated = np.tile(mos.numpy(), (proj_A.shape[0] // mos.shape[0]))
            mos_repeated = mos_repeated[:proj_A.shape[0]]  # Trim to match proj_A size

            predictions.append(prediction)
            mos_scores.append(mos_repeated)

    mos_scores = np.hstack(mos_scores)
    predictions = np.hstack(predictions)

    # Debug: 최종 크기 확인
    print(f"Final mos_scores shape: {mos_scores.shape}, predictions shape: {predictions.shape}")
    return mos_scores, predictions



def plot_results(mos_scores, predictions):
    print(f"Plotting results: mos_scores shape = {mos_scores.shape}, predictions shape = {predictions.shape}")
    assert mos_scores.shape == predictions.shape, "mos_scores and predictions must have the same shape"

    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Ridge Regressor Performance')
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == "__main__":

    # Config 경로 설정
    config_path = "E:/ARNIQA - SE/ARNIQA/config.yaml"
    config = load_config(config_path)

    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(config.data_base_path) / "kadid10k.csv"
    dataset = KADID10KDataset(dataset_path)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=min(config.training.num_workers, 16))
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=min(config.training.num_workers, 16))

    model = SimCLR(encoder_params=DotMap(config.model.encoder), temperature=config.model.temperature).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=config.training.optimizer.momentum, weight_decay=config.training.optimizer.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.training.lr_scheduler.T_0, T_mult=config.training.lr_scheduler.T_mult, eta_min=config.training.lr_scheduler.eta_min)
    scaler = torch.amp.GradScaler()

    train(config, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    regressor = train_ridge_regressor(model, train_dataloader, device)
    mos_scores, predictions = evaluate_ridge_regressor(regressor, model, val_dataloader, device)
    plot_results(mos_scores, predictions)

 """

# # kadid ver3 (ridge regressor + 검증) -? srcc&plcc 완 ( 그래프 생성 완 )
# train & ridge regressor에 대한 모든 검증 완완완
# 실행할거면 이거 실행해야됨


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
from sklearn.linear_model import Ridge
from scipy import stats
import yaml
from sklearn.model_selection import GridSearchCV

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def verify_positive_pairs(distortions_A, distortions_B):
    
    # 양성쌍이 동일한 왜곡이 적용되었는지 확인합니다.
    
    if distortions_A == distortions_B:
        print("[Positive Pair Verification] Success: Distortions match.")
    else:
        print("[Positive Pair Verification] Error: Distortions do not match.")
        print(f"distortions_A: {distortions_A}, distortions_B: {distortions_B}")

def verify_hard_negatives(original_shape, downscaled_shape):
    
    # 경음성쌍이 정확히 50%로 다운스케일 되었는지 확인합니다.
    
    expected_shape = (original_shape[-2] // 2, original_shape[-1] // 2)
    if downscaled_shape[-2:] == expected_shape:
        print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
    else:
        print("[Hard Negative Verification] Error: Hard negatives are not correctly downscaled.")
        print(f"Expected: {expected_shape}, Got: {downscaled_shape[-2:]}")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    
    # Ensure shapes are consistent
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"

    # Flatten both vectors for overall SRCC/PLCC
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    
    return srocc, plcc



def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)

            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            proj_A, proj_B = model(inputs_A, inputs_B)
            srocc, _ = stats.spearmanr(proj_A.flatten().cpu(), proj_B.flatten().cpu())
            plcc, _ = stats.pearsonr(proj_A.flatten().cpu(), proj_B.flatten().cpu())

            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.StepLR,
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:
    # 수정된 checkpoint_path
    checkpoint_path = Path(str(args.checkpoint_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # 동일한 왜곡을 적용하도록 shared_distortion 설정
            shared_distortion = random.choice(
                ["gaussian_blur", "lens_blur", "motion_blur", "color_diffusion", "color_shift", "color_quantization", "color_saturation_1", "color_saturation_2",
                 "jpeg2000", "jpeg", "white_noise", "white_noise_color_component", "impulse_noise", "multiplicative_noise", "denoise", "brighten", "darken",
                 "mean_shift", "jitter", "non_eccentricity_patch", "pixelate", "quantization", "color_block", "high_sharpen", "contrast_change", "blur", "noise"])
                 
            inputs_A, distortions_A = apply_random_distortions(inputs_A, shared_distortion=shared_distortion, return_info=True)
            inputs_B, distortions_B = apply_random_distortions(inputs_B, shared_distortion=shared_distortion, return_info=True)

            # 양성 쌍 검증
            if distortions_A == distortions_B:
                print("[Positive Pair Verification] Success: Distortions match.")
            else:
                print(f"[Positive Pair Verification] Error: Distortions do not match.\n"
                      f"distortions_A: {distortions_A}, distortions_B: {distortions_B}")

            # 입력 데이터가 5차원([batch_size, C, H, W])인 경우에만 unsqueeze 호출
            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)

            # num_crops 차원을 2로 확장
            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            # 경음성 쌍 생성 및 검증
            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)
            if inputs_B.shape[0] == hard_negatives.shape[0]:
                print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
            else:
                print(f"[Hard Negative Verification] Error: Shape mismatch.\n"
                      f"inputs_B shape: {inputs_B.shape}, hard_negatives shape: {hard_negatives.shape}")

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B, hard_negatives)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        lr_scheduler.step()
        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}] Validation Results: SRCC = {avg_srocc_val:.4f}, PLCC = {avg_plcc_val:.4f}")

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

    print("Finished training")

def optimize_ridge_alpha(embeddings, mos_scores):
    
    # Ridge Regressor의 alpha 값을 최적화합니다.
    
    from sklearn.model_selection import GridSearchCV

    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, scoring='r2', cv=5)
    grid.fit(embeddings, mos_scores)

    best_alpha = grid.best_params_['alpha']
    print(f"Optimal alpha: {best_alpha}")
    return Ridge(alpha=best_alpha).fit(embeddings, mos_scores)


def train_ridge_regressor(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_dataloader):
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)

            proj_A, _ = model(inputs_A, inputs_A)

            # Debug: Shape 확인
            print(f"[Batch {batch_idx}] proj_A shape: {proj_A.shape}, mos shape: {mos.shape}")

            # MOS 점수 반복하여 임베딩과 크기 맞추기
            repeat_factor = proj_A.shape[0] // mos.shape[0]
            mos_repeated = np.repeat(mos.numpy(), repeat_factor)[:proj_A.shape[0]]

            embeddings.append(proj_A.cpu().numpy())
            mos_scores.append(mos_repeated)

    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)

    # Debug: 최종 크기 출력
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Final MOS scores shape: {mos_scores.shape}")

    # Ridge Regressor 최적 alpha 탐색
    regressor = optimize_ridge_alpha(embeddings, mos_scores)
    return regressor

def evaluate_ridge_regressor(regressor, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
                inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)

            proj_A, _ = model(inputs_A, inputs_A)
            prediction = regressor.predict(proj_A.cpu().numpy())

            # MOS 점수 크기 맞추기
            repeat_factor = proj_A.shape[0] // mos.shape[0]
            mos_repeated = np.repeat(mos.numpy(), repeat_factor)[:proj_A.shape[0]]

            predictions.append(prediction)
            mos_scores.append(mos_repeated)

    mos_scores = np.hstack(mos_scores)
    predictions = np.hstack(predictions)

    # Debug: 최종 크기 확인
    print(f"Final mos_scores shape: {mos_scores.shape}, predictions shape: {predictions.shape}")
    return mos_scores, predictions


def plot_results(mos_scores, predictions):
    print(f"Plotting results: mos_scores shape = {mos_scores.shape}, predictions shape = {predictions.shape}")
    assert mos_scores.shape == predictions.shape, "mos_scores and predictions must have the same shape"

    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title('Ridge Regressor Performance')
    plt.legend()
    plt.grid()
    plt.show()
    

if __name__ == "__main__":
    # Config 경로 설정
    config_path = "E:/ARNIQA - SE/ARNIQA/config.yaml"
    config = load_config(config_path)

    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(config.data_base_path) / "kadid10k.csv"
    dataset = KADID10KDataset(dataset_path)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=min(config.training.num_workers, 16),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=min(config.training.num_workers, 16),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=min(config.training.num_workers, 16),
    )

    # 모델, 옵티마이저, 스케줄러 초기화
    model = SimCLR(encoder_params=DotMap(config.model.encoder), temperature=config.model.temperature).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.training.learning_rate,
        momentum=config.training.optimizer.momentum,
        weight_decay=config.training.optimizer.weight_decay,
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.training.lr_scheduler.T_0,
        T_mult=config.training.lr_scheduler.T_mult,
        eta_min=config.training.lr_scheduler.eta_min,
    )
    scaler = torch.amp.GradScaler()

    # 모델 학습
    train(config, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    # Ridge Regressor 학습 (Train 데이터 사용)
    regressor = train_ridge_regressor(model, train_dataloader, device)

    # Validation 데이터에서 Ridge Regressor 평가
    val_mos_scores, val_predictions = evaluate_ridge_regressor(regressor, model, val_dataloader, device)
    val_srcc, _ = stats.spearmanr(val_mos_scores, val_predictions)
    val_plcc, _ = stats.pearsonr(val_mos_scores, val_predictions)

    # Test 데이터에서 Ridge Regressor 평가
    test_mos_scores, test_predictions = evaluate_ridge_regressor(regressor, model, test_dataloader, device)
    test_srcc, _ = stats.spearmanr(test_mos_scores, test_predictions)
    test_plcc, _ = stats.pearsonr(test_mos_scores, test_predictions)

    # 최종 결과 출력
    print(f"\nFinal Validation Results: SRCC = {val_srcc:.4f}, PLCC = {val_plcc:.4f}")
    print(f"Final Test Results: SRCC = {test_srcc:.4f}, PLCC = {test_plcc:.4f}")

    # 그래프 출력 (Test 결과)
    plot_results(test_mos_scores, test_predictions)


# Final embeddings shape: (9974, 2048)
# Final MOS scores shape: (9974,)
# Optimal alpha: 0.01


# kadid ver4 (regressor -> random forest)
# 그냥 해본 거 (의미x)
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils import parse_config
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
import matplotlib.pyplot as plt
import random
from typing import Tuple
import argparse
from sklearn.linear_model import Ridge
from scipy import stats
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time

# Config loader
def load_config(config_path: str) -> DotMap:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return DotMap(config)

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def verify_positive_pairs(distortions_A, distortions_B):
    
    # 양성쌍이 동일한 왜곡이 적용되었는지 확인합니다.
    
    if distortions_A == distortions_B:
        print("[Positive Pair Verification] Success: Distortions match.")
    else:
        print("[Positive Pair Verification] Error: Distortions do not match.")
        print(f"distortions_A: {distortions_A}, distortions_B: {distortions_B}")

def verify_hard_negatives(original_shape, downscaled_shape):
    
    # 경음성쌍이 정확히 50%로 다운스케일 되었는지 확인합니다.
    
    expected_shape = (original_shape[-2] // 2, original_shape[-1] // 2)
    if downscaled_shape[-2:] == expected_shape:
        print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
    else:
        print("[Hard Negative Verification] Error: Hard negatives are not correctly downscaled.")
        print(f"Expected: {expected_shape}, Got: {downscaled_shape[-2:]}")

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    
    # Ensure shapes are consistent
    assert proj_A.shape == proj_B.shape, "Shape mismatch between proj_A and proj_B"

    # Flatten both vectors for overall SRCC/PLCC
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    
    return srocc, plcc



def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)

            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            proj_A, proj_B = model(inputs_A, inputs_B)
            srocc, _ = stats.spearmanr(proj_A.flatten().cpu(), proj_B.flatten().cpu())
            plcc, _ = stats.pearsonr(proj_A.flatten().cpu(), proj_B.flatten().cpu())

            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc

def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.StepLR,
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:
    # 수정된 checkpoint_path
    checkpoint_path = Path(str(args.checkpoint_path))
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # 동일한 왜곡을 적용하도록 shared_distortion 설정
            shared_distortion = random.choice(
                ["gaussian_blur", "lens_blur", "motion_blur", "color_diffusion", "color_shift", "color_quantization", "color_saturation_1", "color_saturation_2",
                 "jpeg2000", "jpeg", "white_noise", "white_noise_color_component", "impulse_noise", "multiplicative_noise", "denoise", "brighten", "darken",
                 "mean_shift", "jitter", "non_eccentricity_patch", "pixelate", "quantization", "color_block", "high_sharpen", "contrast_change", "blur", "noise"])
                 
            inputs_A, distortions_A = apply_random_distortions(inputs_A, shared_distortion=shared_distortion, return_info=True)
            inputs_B, distortions_B = apply_random_distortions(inputs_B, shared_distortion=shared_distortion, return_info=True)

            # 양성 쌍 검증
            if distortions_A == distortions_B:
                print("[Positive Pair Verification] Success: Distortions match.")
            else:
                print(f"[Positive Pair Verification] Error: Distortions do not match.\n"
                      f"distortions_A: {distortions_A}, distortions_B: {distortions_B}")

            # 입력 데이터가 5차원([batch_size, C, H, W])인 경우에만 unsqueeze 호출
            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)

            # num_crops 차원을 2로 확장
            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            # 경음성 쌍 생성 및 검증
            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)
            if inputs_B.shape[0] == hard_negatives.shape[0]:
                print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")
            else:
                print(f"[Hard Negative Verification] Error: Shape mismatch.\n"
                      f"inputs_B shape: {inputs_B.shape}, hard_negatives shape: {hard_negatives.shape}")

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B, hard_negatives)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        lr_scheduler.step()
        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}] Validation Results: SRCC = {avg_srocc_val:.4f}, PLCC = {avg_plcc_val:.4f}")

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

    print("Finished training")

def optimize_ridge_alpha(embeddings, mos_scores):
    
    # Ridge Regressor의 alpha 값을 최적화합니다.
    
    from sklearn.model_selection import GridSearchCV

    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    grid = GridSearchCV(ridge, param_grid, scoring='r2', cv=5)
    grid.fit(embeddings, mos_scores)

    best_alpha = grid.best_params_['alpha']
    print(f"Optimal alpha: {best_alpha}")
    return Ridge(alpha=best_alpha).fit(embeddings, mos_scores)


def train_random_forest(model: nn.Module, train_dataloader: DataLoader, device: torch.device):
    model.eval()
    embeddings, mos_scores = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(train_dataloader):
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)

            proj_A, _ = model(inputs_A, inputs_A)

            # Debug: Shape 확인
            print(f"[Batch {batch_idx}] proj_A shape: {proj_A.shape}, mos shape: {mos.shape}")

            # MOS 점수 반복하여 임베딩과 크기 맞추기
            repeat_factor = proj_A.shape[0] // mos.shape[0]
            mos_repeated = np.repeat(mos.numpy(), repeat_factor)[:proj_A.shape[0]]

            embeddings.append(proj_A.cpu().numpy())
            mos_scores.append(mos_repeated)

    embeddings = np.vstack(embeddings)
    mos_scores = np.hstack(mos_scores)

    # Debug: 최종 크기 출력
    print(f"Final embeddings shape: {embeddings.shape}")
    print(f"Final MOS scores shape: {mos_scores.shape}")

    # Random Forest 학습 시작 로그
    print("Defining the Random Forest model...")
    regressor = RandomForestRegressor(
        n_estimators=10,  # 트리 개수
        max_depth=10,      # 최대 깊이
        n_jobs=-1,         # 병렬 처리
        random_state=42
    )

    print("Starting Random Forest training...")
    start_time = time.time()
    regressor.fit(embeddings, mos_scores)  # 학습
    print(f"Random Forest training completed in {time.time() - start_time:.2f} seconds.")
    
    return regressor


def evaluate_random_forest(regressor, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
                inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)

            proj_A, _ = model(inputs_A, inputs_A)
            prediction = regressor.predict(proj_A.cpu().numpy())

            # MOS 점수 크기 맞추기
            repeat_factor = proj_A.shape[0] // mos.shape[0]
            mos_repeated = np.repeat(mos.numpy(), repeat_factor)[:proj_A.shape[0]]

            predictions.append(prediction)
            mos_scores.append(mos_repeated)

    mos_scores = np.hstack(mos_scores)
    predictions = np.hstack(predictions)

    # Debug: 최종 크기 확인
    print(f"Final mos_scores shape: {mos_scores.shape}, predictions shape: {predictions.shape}")
    return mos_scores, predictions


def plot_results_with_metrics(mos_scores, predictions, srcc, plcc):
    print(f"Plotting results: mos_scores shape = {mos_scores.shape}, predictions shape = {predictions.shape}")
    assert mos_scores.shape == predictions.shape, "mos_scores and predictions must have the same shape"

    plt.figure(figsize=(8, 6))
    plt.scatter(mos_scores, predictions, alpha=0.7, label='Predictions vs MOS')
    plt.plot([min(mos_scores), max(mos_scores)], [min(mos_scores), max(mos_scores)], 'r--', label='Ideal')
    plt.xlabel('Ground Truth MOS')
    plt.ylabel('Predicted MOS')
    plt.title(f'Random Forest Performance (SRCC: {srcc:.4f}, PLCC: {plcc:.4f})')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Config 경로 설정
    config_path = "E:/ARNIQA - SE/ARNIQA/config.yaml"
    config = load_config(config_path)

    device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
    dataset_path = Path(config.data_base_path) / "kadid10k.csv"
    dataset = KADID10KDataset(dataset_path)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True, num_workers=min(config.training.num_workers, 16))
    val_dataloader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=min(config.training.num_workers, 16))
    test_dataloader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=min(config.training.num_workers, 16))

    model = SimCLR(encoder_params=DotMap(config.model.encoder), temperature=config.model.temperature).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=config.training.optimizer.momentum, weight_decay=config.training.optimizer.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.training.lr_scheduler.T_0, T_mult=config.training.lr_scheduler.T_mult, eta_min=config.training.lr_scheduler.eta_min)
    scaler = torch.amp.GradScaler()

    # Training
    train(config, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

    # Train Random Forest
    regressor = train_random_forest(model, train_dataloader, device)

    # Evaluate Random Forest on Test Data
    mos_scores, predictions = evaluate_random_forest(regressor, model, test_dataloader, device)

    # Calculate SRCC and PLCC
    srcc, _ = stats.spearmanr(mos_scores, predictions)
    plcc, _ = stats.pearsonr(mos_scores, predictions)

    print(f"Final Test Results: SRCC = {srcc:.4f}, PLCC = {plcc:.4f}")

    # Plot results with metrics
    plot_results_with_metrics(mos_scores, predictions, srcc, plcc)

 """

# ---------------------------------------------------------------#
# 이 아래로 수정 안됨
# 실행 x


# kadid ver2 (ridge regressor + 검증)
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
from torch.cuda.amp import custom_fwd
import matplotlib.pyplot as plt


def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)


def verify_positive_pairs(inputs_A, inputs_B, distortions_A, distortions_B):
    if distortions_A != distortions_B:
        print("[Positive Pair Verification] Error: Distortions do not match.")
        print(f"distortions_A: {distortions_A}, distortions_B: {distortions_B}")
    else:
        print("[Positive Pair Verification] Success: Distortions match.")


def verify_hard_negatives(inputs_B, hard_negatives):
    original_shape = inputs_B.shape[-2:]  # H, W
    hard_negative_shape = hard_negatives.shape[-2:]
    expected_shape = (original_shape[0] // 2, original_shape[1] // 2)
    if hard_negative_shape != expected_shape:
        print(f"[Hard Negative Verification] Error: Expected {expected_shape}, but got {hard_negative_shape}.")
    else:
        print("[Hard Negative Verification] Success: Hard negatives are correctly downscaled.")


def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


def validate(args, model, val_dataloader, device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)

            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            proj_A, proj_B = model(inputs_A, inputs_B)
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)

            srocc_list.append(srocc)
            plcc_list.append(plcc)

    return np.mean(srocc_list), np.mean(plcc_list)


def train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device):
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # 동일한 왜곡 적용
            shared_distortion = torch.randint(0, 4, (1,)).item()  # 동일한 왜곡 생성
            inputs_A, distortions_A = apply_random_distortions(inputs_A, shared_distortion=shared_distortion, return_info=True)
            inputs_B, distortions_B = apply_random_distortions(inputs_B, shared_distortion=shared_distortion, return_info=True)

            # 양성 쌍 검증
            verify_positive_pairs(inputs_A, inputs_B, distortions_A, distortions_B)

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
            if inputs_B.dim() == 4:
                inputs_B = inputs_B.unsqueeze(1)

            inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)
            inputs_B = inputs_B.expand(-1, 2, -1, -1, -1)

            # 경음성 쌍 생성 및 검증
            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)
            verify_hard_negatives(inputs_B, hard_negatives)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B, hard_negatives)
                print(f"[Loss Debug] Loss: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (i + 1))

        lr_scheduler.step()
        avg_srocc, avg_plcc = validate(args, model, val_dataloader, device)
        print(f"Validation SRCC: {avg_srocc}, PLCC: {avg_plcc}")

        if avg_srocc > best_srocc:
            best_srocc = avg_srocc
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

    print("Training Complete.")


if __name__ == "__main__":
    args = DotMap({
        "training": {"epochs": 10, "batch_size": 16, "learning_rate": 1e-3, "num_workers": 4},
        "checkpoint_path": "E:/ARNIQA/experiments/my_experiment/pretrain",
        "model": {"temperature": 0.1, "embedding_dim": 128}
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KADID10KDataset("E:/ARNIQA - SE/ARNIQA/dataset/KADID10K/kadid10k.csv")
    train_size, val_size = int(0.7 * len(dataset)), int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False)

    model = SimCLR(
        encoder_params=DotMap({"embedding_dim": args.model.embedding_dim}),
        temperature=args.model.temperature
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.cuda.amp.GradScaler()

    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

 """

# ------------------------------------
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np
from dotmap import DotMap
from pathlib import Path
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import Ridge
from data import KADID10KDataset
from models.simclr import SimCLR
from utils.utils_distortions import apply_random_distortions, generate_hard_negatives
from torch.cuda.amp import custom_fwd


def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)


def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())
    return srocc, plcc


@custom_fwd(cast_inputs=torch.float32)
def validate(args: DotMap, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    srocc_list, plcc_list = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Apply same distortions to both inputs for positive pairs
            inputs_A = apply_random_distortions(inputs_A)
            inputs_B = inputs_A.clone()

            # Reshape inputs to match SimCLR requirements
            inputs_A = inputs_A.unsqueeze(1)
            inputs_B = inputs_B.unsqueeze(1)

            proj_A, proj_B = model(inputs_A, inputs_B)
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)

            srocc_list.append(srocc)
            plcc_list.append(plcc)

    avg_srocc = np.mean(srocc_list) if srocc_list else 0
    avg_plcc = np.mean(plcc_list) if plcc_list else 0
    return avg_srocc, avg_plcc


def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          val_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.StepLR,
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:
    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_srocc = 0
    for epoch in range(args.training.epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{args.training.epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A = batch["img_A"].to(device)
            inputs_B = batch["img_B"].to(device)

            # Apply same distortions for positive pairs
            inputs_A = apply_random_distortions(inputs_A)
            inputs_B = inputs_A.clone()

            # Reshape inputs to match SimCLR requirements
            inputs_A = inputs_A.unsqueeze(1)
            inputs_B = inputs_B.unsqueeze(1)

            # Generate hard negatives
            hard_negatives = generate_hard_negatives(inputs_B, scale_factor=0.5)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B, hard_negatives)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        lr_scheduler.step()

        avg_srocc_val, avg_plcc_val = validate(args, model, val_dataloader, device)
        print(f"Epoch [{epoch + 1}] Validation Results: SRCC = {avg_srocc_val:.4f}, PLCC = {avg_plcc_val:.4f}")

        if avg_srocc_val > best_srocc:
            best_srocc = avg_srocc_val
            save_checkpoint(model, checkpoint_path, epoch, best_srocc)

    print("Finished training")


if __name__ == "__main__":
    args = DotMap({
        "data_base_path": "E:/ARNIQA/dataset",
        "training": {
            "epochs": 30,
            "batch_size": 32,
            "learning_rate": 1e-4,
            "num_workers": 4,
        },
        "checkpoint_path": "E:/ARNIQA/experiments/my_experiment/pretrain",
        "model": {
            "temperature": 0.5,
            "embedding_dim": 128
        }
    })

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = KADID10KDataset("E:/ARNIQA - SE/ARNIQA/dataset/KADID10K/kadid10k.csv")
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.training.batch_size * 2,
        shuffle=True,
        num_workers=args.training.num_workers,
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.training.batch_size, shuffle=False, num_workers=args.training.num_workers)

    model = SimCLR(encoder_params=DotMap({"embedding_dim": args.model.embedding_dim, "pretrained": True}), temperature=args.model.temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scaler = torch.amp.GradScaler()

    train(args, model, train_dataloader, val_dataloader, optimizer, lr_scheduler, scaler, device)

 """

#tid2013

""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import openpyxl
import pandas
from openpyxl.styles import Alignment
import pickle
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
import argparse
from tqdm import tqdm
from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR

synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "spaq"]

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def calculate_srcc_plcc(proj_A, proj_B):
    # 모델 출력값을 넘파이 배열로 변환
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()

    # SRCC 계산
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())

    # PLCC 계산
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

    return srocc, plcc



def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:

    # 수정된 체크포인트 경로
    checkpoint_path = Path('E:/ARNIQA - SE/ARNIQA/experiments/my_experiment/pretrain3_tid')
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoints in folder: ", checkpoint_path)

    start_epoch = 0
    max_epochs = args.training.epochs
    best_srocc = 0

    # SRCC와 PLCC 기록 리스트
    all_srocc_values = []
    all_plcc_values = []

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        epoch_diff = 0.0  # 에포크 동안의 차이 누적값
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{max_epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A_orig = batch["img_A_orig"].to(device=device, non_blocking=True)
            inputs_A_ds = batch["img_A_ds"].to(device=device, non_blocking=True)

            # Concatenate along the batch dimension and remove the extra dimension
            inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=1)
            inputs_A = inputs_A.view(-1, 4, 3, 224, 224)  # Flatten to [batch_size * 2, num_crops, C, H, W]

            inputs_B_orig = batch["img_A_ds"].to(device=device, non_blocking=True)
            inputs_B_ds = batch["img_B_ds"].to(device=device, non_blocking=True)

            inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=1)
            inputs_B = inputs_B.view(-1, 4, 3, 224, 224)  # Flatten to [batch_size * 2, num_crops, C, H, W]

            print(f"Adjusted inputs_A shape: {inputs_A.shape}, inputs_B shape: {inputs_B.shape}")

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            with torch.amp.autocast(device_type='cuda'):
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cur_loss = loss.item()
            running_loss += cur_loss

            # SRCC 및 PLCC 계산
            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)

            # proj_A와 proj_B의 차이(평균) 계산
            diff = torch.mean(torch.abs(proj_A - proj_B))  # 차이의 평균을 계산
            epoch_diff += diff.item()  # 에포크 동안 차이 누적

            # SRCC와 PLCC 기록
            all_srocc_values.append(srocc)
            all_plcc_values.append(plcc)

            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        # 에포크 끝날 때 평균 차이 출력
        avg_diff_epoch = epoch_diff / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{max_epochs}] - Average difference between proj_A and proj_B: {avg_diff_epoch:.4f}")

        # Save checkpoints at regular intervals
        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

    # 훈련이 끝난 후 최종 SRCC, PLCC 출력
    final_srocc = np.mean(all_srocc_values)
    final_plcc = np.mean(all_plcc_values)

    print(f"Final SRCC: {final_srocc:.4f}, Final PLCC: {final_plcc:.4f}")
    print('Finished training')

def validate(args: DotMap,
             model: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    model.eval()
    
    # KADID10K 데이터셋 및 SPAQ 데이터셋 사용
    datasets = ['TID2013']
    for dataset_name in datasets:
        print(f"Validating dataset: {dataset_name}")

    srocc_all, plcc_all, _, _, _ = get_results(model=model, data_base_path=args.data_base_path,
                                               datasets=datasets,
                                               num_splits=args.validation.num_splits,
                                               phase="val", alpha=args.validation.alpha, grid_search=False,
                                               crop_size=args.test.crop_size, batch_size=args.test.batch_size,
                                               num_workers=args.test.num_workers, device=device)

    # Compute the median for each list in srocc_all and plcc_all
    srocc_all_median = {key: np.median(value["global"]) for key, value in srocc_all.items()}
    plcc_all_median = {key: np.median(value["global"]) for key, value in plcc_all.items()}

    # Compute the global average
    srocc_avg = np.mean(list(srocc_all_median.values()))
    plcc_avg = np.mean(list(plcc_all_median.values()))

    return srocc_avg, plcc_avg


def get_results(model: nn.Module,
                data_base_path: Path,
                datasets: List[str],
                num_splits: int,
                phase: str,
                alpha: float,
                grid_search: bool,
                crop_size: int,
                batch_size: int,
                num_workers: int,
                device: torch.device,
                eval_type: str = "scratch") -> Tuple[dict, dict, dict, dict, dict]:
    srocc_all = {}
    plcc_all = {}
    regressors = {}
    alphas = {}
    best_worst_results_all = {}

    assert phase in ["val", "test"], "Phase must be in ['val', 'test']"

    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Starting {phase} phase")
    for d in datasets:
        if d == "live":
            dataset = LIVEDataset(data_base_path / "LIVE", phase="all", crop_size=crop_size)
        elif d == "csiq":
            dataset = CSIQDataset(data_base_path / "CSIQ", phase="all", crop_size=crop_size)
        elif d == "tid2013":
            dataset = TID2013Dataset(data_base_path / "TID2013", phase="all", crop_size=crop_size)
        elif d == "kadid10k":
            dataset = KADID10KDataset(data_base_path / "KADID10K", phase="all", crop_size=crop_size)
        elif d == "flive":
            dataset = FLIVEDataset(data_base_path / "FLIVE", phase="all", crop_size=crop_size)
        elif d == "spaq":
            dataset = SPAQDataset(data_base_path / "SPAQ", phase="all", crop_size=crop_size)
        else:
            raise ValueError(f"Dataset {d} not supported")

        # 결과 계산
        srocc_dataset, plcc_dataset, regressor, alpha_value, best_worst_results = compute_metrics(model, dataset,
                                                                                                num_splits, phase,
                                                                                                alpha, grid_search,
                                                                                                batch_size, num_workers,
                                                                                                device, eval_type)
        srocc_all[d] = srocc_dataset
        plcc_all[d] = plcc_dataset
        regressors[d] = regressor
        alphas[d] = alpha_value
        best_worst_results_all[d] = best_worst_results
        print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {d}:" f" SRCC: {np.median(srocc_dataset['global']):.3f} - PLCC: {np.median(plcc_dataset['global']):.3f}")

    return srocc_all, plcc_all, regressors, alphas, best_worst_results_all


def compute_metrics(model: nn.Module,
                    dataset: DataLoader,
                    num_splits: int,
                    phase: str,
                    alpha: float,
                    grid_search: bool,
                    batch_size: int,
                    num_workers: int,
                    device: torch.device,
                    eval_type: str = "scratch") -> Tuple[dict, dict, Ridge, float, dict]:
    srocc_dataset = {"global": []}
    plcc_dataset = {"global": []}
    best_worst_results = {}

    # DataLoader 설정
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # features 및 scores 가져오기
    features, scores = get_features_scores(model, dataloader, device, eval_type)

    # Debugging: features와 scores의 첫 10개 값을 확인
    print(f"Features: {features[:10]}")
    print(f"Scores: {scores[:10]}")

    # Grid search 또는 alpha 값을 사용하여 회귀 모델 학습
    if phase == "test" and grid_search:
        best_alpha = alpha_grid_search(dataset=dataset, features=features, scores=scores, num_splits=num_splits)
    else:
        best_alpha = alpha

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        test_indices = dataset.get_split_indices(split=i, phase=phase)

        # Train features 및 scores 가져오기
        train_features = features[train_indices]
        train_scores = scores[train_indices]

        # 회귀 모델 학습
        regressor = Ridge(alpha=best_alpha).fit(train_features, train_scores)

        # Test features 및 scores 가져오기
        test_features = features[test_indices]
        test_scores = scores[test_indices]

        # 예측 수행
        preds = regressor.predict(test_features)
        preds = preds.flatten()

        # Debugging: 예측 값 및 실제 라벨 확인
        print(f"Predictions: {preds[:10]}")
        print(f"Test Scores: {test_scores.flatten()[:10]}")

        # SROCC 및 PLCC 계산
        srocc_value = stats.spearmanr(preds, test_scores.flatten())[0]
        plcc_value = stats.pearsonr(preds, test_scores.flatten())[0]
        print(f"SROCC: {srocc_value}, PLCC: {plcc_value}")

        srocc_dataset["global"].append(srocc_value)
        plcc_dataset["global"].append(plcc_value)

    return srocc_dataset, plcc_dataset, regressor, best_alpha, best_worst_results


def get_features_scores(model, dataloader, device, eval_type):
    scores = np.array([])  # 초기화
    mos = np.array([])  # 초기화

    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for i, batch in enumerate(dataloader):
            print(f"Batch {i} keys: {batch.keys()}")

            # Check if the expected keys are present in the batch
            if not all(key in batch for key in ['img_A_orig', 'img_B_orig']):
                print(f"Missing keys in batch {i}: {[key for key in ['img_A_orig', 'img_B_orig'] if key not in batch]}")
                continue

            print(f"Batch {i} mos: {batch['mos']}")  # Debugging: 'mos'의 내용 확인

            # Convert 'mos' to numpy array if it is a list
            if isinstance(batch['mos'], list):
                mos_batch = np.array(batch['mos'])
            else:
                mos_batch = batch['mos'].cpu().numpy()  # Ensure it is on CPU and convert to numpy

            mos = np.concatenate((mos, mos_batch), axis=0)  # Concatenate the mos

            # 이미지 데이터 가져오기
            img_A_orig = batch["img_A_orig"].to(device)
            img_B_orig = batch["img_B_orig"].to(device)

            # Check shapes
            print(f"img_A_orig shape: {img_A_orig.shape}, img_B_orig shape: {img_B_orig.shape}")  # Shape 확인

            # 모델에 대한 피처 추출
            with torch.amp.autocast(device_type='cuda'):
                feature_A, feature_B = model(img_A_orig, img_B_orig)  # 모델에서 두 개의 피처를 얻습니다.

def alpha_grid_search(dataset: Dataset,
                      features: np.ndarray,
                      scores: np.ndarray,
                      num_splits: int) -> float:


    grid_search_range = [1e-3, 1e3, 100]
    alphas = np.geomspace(*grid_search_range, endpoint=True)
    srocc_all = [[] for _ in range(len(alphas))]

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        val_indices = dataset.get_split_indices(split=i, phase="val")

        # for each index generate 5 indices (one for each crop)
        train_indices = np.repeat(train_indices * 5, 5) + np.tile(np.arange(5), len(train_indices))
        val_indices = np.repeat(val_indices * 5, 5) + np.tile(np.arange(5), len(val_indices))

        train_features = features[train_indices]
        train_scores = scores[train_indices]

        val_features = features[val_indices]
        val_scores = scores[val_indices]
        val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one

        for idx, alpha in enumerate(alphas):
            regressor = Ridge(alpha=alpha).fit(train_features, train_scores)
            preds = regressor.predict(val_features)
            preds = np.mean(np.reshape(preds, (-1, 5)), 1)  # Average the predictions of the 5 crops of the same image
            srocc_all[idx].append(stats.spearmanr(preds, val_scores)[0])

    srocc_all_median = [np.median(srocc) for srocc in srocc_all]
    srocc_all_median = np.array(srocc_all_median)
    best_alpha_idx = np.argmax(srocc_all_median)
    best_alpha = alphas[best_alpha_idx]

    return best_alpha



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    args, _ = parser.parse_known_args()
    config = parse_config(args.config)
    args = parse_command_line_args(config)
    args = merge_configs(config, args)

    args.data_base_path = Path(args.data_base_path)  # 경로를 Path 객체로 변환
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로더 설정
    train_dataset = TID2013Dataset(Path('E:/ARNIQA/ARNIQA/dataset/TID2013'), phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)

    # Optimizer 및 모델 초기화
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.amp.GradScaler()

    # 훈련 시작
    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)
 """



# kadid train & tid Test
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import openpyxl
import pandas
from openpyxl.styles import Alignment
import pickle
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
import argparse
from tqdm import tqdm
from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR

synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "spaq"]

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def calculate_srcc_plcc(proj_A, proj_B):
    # 모델 출력값을 넘파이 배열로 변환
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()

    # SRCC 계산
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())

    # PLCC 계산
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

    return srocc, plcc



def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:

    checkpoint_path = Path('E:/ARNIQA - SE/ARNIQA/experiments/my_experiment/pretrain4_kadid_tid')
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoints in folder: ", checkpoint_path)

    start_epoch = 0
    max_epochs = args.training.epochs
    best_srocc = 0

    # SRCC와 PLCC 기록 리스트
    all_srocc_values = []
    all_plcc_values = []

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        epoch_diff = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{max_epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A_orig = batch["img_A_orig"].to(device=device, non_blocking=True)
            inputs_A_ds = batch["img_A_ds"].to(device=device, non_blocking=True)

            # Concatenate along the batch dimension and remove the extra dimension
            inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=1)
            inputs_A = inputs_A.view(-1, 4, 3, 224, 224)

            inputs_B_orig = batch["img_A_ds"].to(device=device, non_blocking=True)
            inputs_B_ds = batch["img_B_ds"].to(device=device, non_blocking=True)

            inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=1)
            inputs_B = inputs_B.view(-1, 4, 3, 224, 224)

            print(f"Adjusted inputs_A shape: {inputs_A.shape}, inputs_B shape: {inputs_B.shape}")

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cur_loss = loss.item()
            running_loss += cur_loss

            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)

            diff = torch.mean(torch.abs(proj_A - proj_B))
            epoch_diff += diff.item()

            all_srocc_values.append(srocc)
            all_plcc_values.append(plcc)

            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        avg_diff_epoch = epoch_diff / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{max_epochs}] - Average difference between proj_A and proj_B: {avg_diff_epoch:.4f}")

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

    final_srocc = np.mean(all_srocc_values)
    final_plcc = np.mean(all_plcc_values)

    print(f"Final SRCC: {final_srocc:.4f}, Final PLCC: {final_plcc:.4f}")
    print('Finished training')

def validate(args: DotMap,
             model: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    model.eval()

    datasets = ['TID2013']  # Test with TID2013 dataset
    for dataset_name in datasets:
        print(f"Validating dataset: {dataset_name}")

    srocc_all, plcc_all, _, _, _ = get_results(model=model, data_base_path=args.data_base_path,
                                               datasets=datasets,
                                               num_splits=args.validation.num_splits,
                                               phase="val", alpha=args.validation.alpha, grid_search=False,
                                               crop_size=args.test.crop_size, batch_size=args.test.batch_size,
                                               num_workers=args.test.num_workers, device=device)

    srocc_all_median = {key: np.median(value["global"]) for key, value in srocc_all.items()}
    plcc_all_median = {key: np.median(value["global"]) for key, value in plcc_all.items()}

    srocc_avg = np.mean(list(srocc_all_median.values()))
    plcc_avg = np.mean(list(plcc_all_median.values()))

    return srocc_avg, plcc_avg
def get_results(model: nn.Module,
                data_base_path: Path,
                datasets: List[str],
                num_splits: int,
                phase: str,
                alpha: float,
                grid_search: bool,
                crop_size: int,
                batch_size: int,
                num_workers: int,
                device: torch.device,
                eval_type: str = "scratch") -> Tuple[dict, dict, dict, dict, dict]:
    srocc_all = {}
    plcc_all = {}
    regressors = {}
    alphas = {}
    best_worst_results_all = {}

    assert phase in ["val", "test"], "Phase must be in ['val', 'test']"

    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Starting {phase} phase")
    for d in datasets:
        if d == "live":
            dataset = LIVEDataset(data_base_path / "LIVE", phase="all", crop_size=crop_size)
        elif d == "csiq":
            dataset = CSIQDataset(data_base_path / "CSIQ", phase="all", crop_size=crop_size)
        elif d == "tid2013":
            dataset = TID2013Dataset(data_base_path / "TID2013", phase="all", crop_size=crop_size)
        elif d == "kadid10k":
            dataset = KADID10KDataset(data_base_path / "KADID10K", phase="all", crop_size=crop_size)
        elif d == "flive":
            dataset = FLIVEDataset(data_base_path / "FLIVE", phase="all", crop_size=crop_size)
        elif d == "spaq":
            dataset = SPAQDataset(data_base_path / "SPAQ", phase="all", crop_size=crop_size)
        else:
            raise ValueError(f"Dataset {d} not supported")

        # 결과 계산
        srocc_dataset, plcc_dataset, regressor, alpha_value, best_worst_results = compute_metrics(model, dataset,
                                                                                                num_splits, phase,
                                                                                                alpha, grid_search,
                                                                                                batch_size, num_workers,
                                                                                                device, eval_type)
        srocc_all[d] = srocc_dataset
        plcc_all[d] = plcc_dataset
        regressors[d] = regressor
        alphas[d] = alpha_value
        best_worst_results_all[d] = best_worst_results
        print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {d}:" f" SRCC: {np.median(srocc_dataset['global']):.3f} - PLCC: {np.median(plcc_dataset['global']):.3f}")

    return srocc_all, plcc_all, regressors, alphas, best_worst_results_all


def compute_metrics(model: nn.Module,
                    dataset: DataLoader,
                    num_splits: int,
                    phase: str,
                    alpha: float,
                    grid_search: bool,
                    batch_size: int,
                    num_workers: int,
                    device: torch.device,
                    eval_type: str = "scratch") -> Tuple[dict, dict, Ridge, float, dict]:
    srocc_dataset = {"global": []}
    plcc_dataset = {"global": []}
    best_worst_results = {}

    # DataLoader 설정
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # features 및 scores 가져오기
    features, scores = get_features_scores(model, dataloader, device, eval_type)

    # Debugging: features와 scores의 첫 10개 값을 확인
    print(f"Features: {features[:10]}")
    print(f"Scores: {scores[:10]}")

    # Grid search 또는 alpha 값을 사용하여 회귀 모델 학습
    if phase == "test" and grid_search:
        best_alpha = alpha_grid_search(dataset=dataset, features=features, scores=scores, num_splits=num_splits)
    else:
        best_alpha = alpha

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        test_indices = dataset.get_split_indices(split=i, phase=phase)

        # Train features 및 scores 가져오기
        train_features = features[train_indices]
        train_scores = scores[train_indices]

        # 회귀 모델 학습
        regressor = Ridge(alpha=best_alpha).fit(train_features, train_scores)

        # Test features 및 scores 가져오기
        test_features = features[test_indices]
        test_scores = scores[test_indices]

        # 예측 수행
        preds = regressor.predict(test_features)
        preds = preds.flatten()

        # Debugging: 예측 값 및 실제 라벨 확인
        print(f"Predictions: {preds[:10]}")
        print(f"Test Scores: {test_scores.flatten()[:10]}")

        # SROCC 및 PLCC 계산
        srocc_value = stats.spearmanr(preds, test_scores.flatten())[0]
        plcc_value = stats.pearsonr(preds, test_scores.flatten())[0]
        print(f"SROCC: {srocc_value}, PLCC: {plcc_value}")

        srocc_dataset["global"].append(srocc_value)
        plcc_dataset["global"].append(plcc_value)

    return srocc_dataset, plcc_dataset, regressor, best_alpha, best_worst_results


def get_features_scores(model, dataloader, device, eval_type):
    scores = np.array([])  # 초기화
    mos = np.array([])  # 초기화

    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for i, batch in enumerate(dataloader):
            print(f"Batch {i} keys: {batch.keys()}")

            # Check if the expected keys are present in the batch
            if not all(key in batch for key in ['img_A_orig', 'img_B_orig']):
                print(f"Missing keys in batch {i}: {[key for key in ['img_A_orig', 'img_B_orig'] if key not in batch]}")
                continue

            print(f"Batch {i} mos: {batch['mos']}")  # Debugging: 'mos'의 내용 확인

            # Convert 'mos' to numpy array if it is a list
            if isinstance(batch['mos'], list):
                mos_batch = np.array(batch['mos'])
            else:
                mos_batch = batch['mos'].cpu().numpy()  # Ensure it is on CPU and convert to numpy

            mos = np.concatenate((mos, mos_batch), axis=0)  # Concatenate the mos

            # 이미지 데이터 가져오기
            img_A_orig = batch["img_A_orig"].to(device)
            img_B_orig = batch["img_B_orig"].to(device)

            # Check shapes
            print(f"img_A_orig shape: {img_A_orig.shape}, img_B_orig shape: {img_B_orig.shape}")  # Shape 확인

            # 모델에 대한 피처 추출
            with torch.amp.autocast(device_type='cuda'):
                feature_A, feature_B = model(img_A_orig, img_B_orig)  # 모델에서 두 개의 피처를 얻습니다.

def alpha_grid_search(dataset: Dataset,
                      features: np.ndarray,
                      scores: np.ndarray,
                      num_splits: int) -> float:


    grid_search_range = [1e-3, 1e3, 100]
    alphas = np.geomspace(*grid_search_range, endpoint=True)
    srocc_all = [[] for _ in range(len(alphas))]

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        val_indices = dataset.get_split_indices(split=i, phase="val")

        # for each index generate 5 indices (one for each crop)
        train_indices = np.repeat(train_indices * 5, 5) + np.tile(np.arange(5), len(train_indices))
        val_indices = np.repeat(val_indices * 5, 5) + np.tile(np.arange(5), len(val_indices))

        train_features = features[train_indices]
        train_scores = scores[train_indices]

        val_features = features[val_indices]
        val_scores = scores[val_indices]
        val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one

        for idx, alpha in enumerate(alphas):
            regressor = Ridge(alpha=alpha).fit(train_features, train_scores)
            preds = regressor.predict(val_features)
            preds = np.mean(np.reshape(preds, (-1, 5)), 1)  # Average the predictions of the 5 crops of the same image
            srocc_all[idx].append(stats.spearmanr(preds, val_scores)[0])

    srocc_all_median = [np.median(srocc) for srocc in srocc_all]
    srocc_all_median = np.array(srocc_all_median)
    best_alpha_idx = np.argmax(srocc_all_median)
    best_alpha = alphas[best_alpha_idx]

    return best_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    args, _ = parser.parse_known_args()
    config = parse_config(args.config)
    args = parse_command_line_args(config)
    args = merge_configs(config, args)

    args.data_base_path = Path(args.data_base_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로더 설정
    train_dataset = KADID10KDataset(Path('E:/ARNIQA/ARNIQA/dataset/KADID10K'), phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)

    # Optimizer 및 모델 초기화
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.amp.GradScaler()

    # 훈련 시작
    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)

 """

# tid train& kadid test
""" import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import openpyxl
import pandas
from openpyxl.styles import Alignment
import pickle
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
import argparse
from tqdm import tqdm
from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR

synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "spaq"]

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def calculate_srcc_plcc(proj_A, proj_B):
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()

    # SRCC 계산
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())

    # PLCC 계산
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

    return srocc, plcc

def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:

    checkpoint_path = Path('E:/ARNIQA - SE/ARNIQA/experiments/my_experiment/pretrain5_tid_kadid')
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoints in folder: ", checkpoint_path)

    start_epoch = 0
    max_epochs = args.training.epochs
    best_srocc = 0

    # SRCC와 PLCC 기록 리스트
    all_srocc_values = []
    all_plcc_values = []

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        epoch_diff = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{max_epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A_orig = batch["img_A_orig"].to(device=device, non_blocking=True)
            inputs_A_ds = batch["img_A_ds"].to(device=device, non_blocking=True)

            # Concatenate along the batch dimension and remove the extra dimension
            inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=1)
            inputs_A = inputs_A.view(-1, 4, 3, 224, 224)

            inputs_B_orig = batch["img_A_ds"].to(device=device, non_blocking=True)
            inputs_B_ds = batch["img_B_ds"].to(device=device, non_blocking=True)

            inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=1)
            inputs_B = inputs_B.view(-1, 4, 3, 224, 224)

            print(f"Adjusted inputs_A shape: {inputs_A.shape}, inputs_B shape: {inputs_B.shape}")

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                proj_A, proj_B = model(inputs_A, inputs_B)
                loss = model.compute_loss(proj_A, proj_B)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cur_loss = loss.item()
            running_loss += cur_loss

            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)

            diff = torch.mean(torch.abs(proj_A - proj_B))
            epoch_diff += diff.item()

            all_srocc_values.append(srocc)
            all_plcc_values.append(plcc)

            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        avg_diff_epoch = epoch_diff / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{max_epochs}] - Average difference between proj_A and proj_B: {avg_diff_epoch:.4f}")

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

    final_srocc = np.mean(all_srocc_values)
    final_plcc = np.mean(all_plcc_values)

    print(f"Final SRCC: {final_srocc:.4f}, Final PLCC: {final_plcc:.4f}")
    print('Finished training')

def validate(args: DotMap,
             model: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    model.eval()

    datasets = ['KADID10K']  # Test with KADID10K dataset
    for dataset_name in datasets:
        print(f"Validating dataset: {dataset_name}")

    srocc_all, plcc_all, _, _, _ = get_results(model=model, data_base_path=args.data_base_path,
                                               datasets=datasets,
                                               num_splits=args.validation.num_splits,
                                               phase="val", alpha=args.validation.alpha, grid_search=False,
                                               crop_size=args.test.crop_size, batch_size=args.test.batch_size,
                                               num_workers=args.test.num_workers, device=device)

    srocc_all_median = {key: np.median(value["global"]) for key, value in srocc_all.items()}
    plcc_all_median = {key: np.median(value["global"]) for key, value in plcc_all.items()}

    srocc_avg = np.mean(list(srocc_all_median.values()))
    plcc_avg = np.mean(list(plcc_all_median.values()))

    return srocc_avg, plcc_avg

def get_results(model: nn.Module,
                data_base_path: Path,
                datasets: List[str],
                num_splits: int,
                phase: str,
                alpha: float,
                grid_search: bool,
                crop_size: int,
                batch_size: int,
                num_workers: int,
                device: torch.device,
                eval_type: str = "scratch") -> Tuple[dict, dict, dict, dict, dict]:
    srocc_all = {}
    plcc_all = {}
    regressors = {}
    alphas = {}
    best_worst_results_all = {}

    assert phase in ["val", "test"], "Phase must be in ['val', 'test']"

    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Starting {phase} phase")
    for d in datasets:
        if d == "live":
            dataset = LIVEDataset(data_base_path / "LIVE", phase="all", crop_size=crop_size)
        elif d == "csiq":
            dataset = CSIQDataset(data_base_path / "CSIQ", phase="all", crop_size=crop_size)
        elif d == "tid2013":
            dataset = TID2013Dataset(data_base_path / "TID2013", phase="all", crop_size=crop_size)
        elif d == "kadid10k":
            dataset = KADID10KDataset(data_base_path / "KADID10K", phase="all", crop_size=crop_size)
        elif d == "flive":
            dataset = FLIVEDataset(data_base_path / "FLIVE", phase="all", crop_size=crop_size)
        elif d == "spaq":
            dataset = SPAQDataset(data_base_path / "SPAQ", phase="all", crop_size=crop_size)
        else:
            raise ValueError(f"Dataset {d} not supported")

        # 결과 계산
        srocc_dataset, plcc_dataset, regressor, alpha_value, best_worst_results = compute_metrics(model, dataset,
                                                                                                num_splits, phase,
                                                                                                alpha, grid_search,
                                                                                                batch_size, num_workers,
                                                                                                device, eval_type)
        srocc_all[d] = srocc_dataset
        plcc_all[d] = plcc_dataset
        regressors[d] = regressor
        alphas[d] = alpha_value
        best_worst_results_all[d] = best_worst_results
        print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {d}:" f" SRCC: {np.median(srocc_dataset['global']):.3f} - PLCC: {np.median(plcc_dataset['global']):.3f}")

    return srocc_all, plcc_all, regressors, alphas, best_worst_results_all


def compute_metrics(model: nn.Module,
                    dataset: DataLoader,
                    num_splits: int,
                    phase: str,
                    alpha: float,
                    grid_search: bool,
                    batch_size: int,
                    num_workers: int,
                    device: torch.device,
                    eval_type: str = "scratch") -> Tuple[dict, dict, Ridge, float, dict]:
    srocc_dataset = {"global": []}
    plcc_dataset = {"global": []}
    best_worst_results = {}

    # DataLoader 설정
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # features 및 scores 가져오기
    features, scores = get_features_scores(model, dataloader, device, eval_type)

    # Debugging: features와 scores의 첫 10개 값을 확인
    print(f"Features: {features[:10]}")
    print(f"Scores: {scores[:10]}")

    # Grid search 또는 alpha 값을 사용하여 회귀 모델 학습
    if phase == "test" and grid_search:
        best_alpha = alpha_grid_search(dataset=dataset, features=features, scores=scores, num_splits=num_splits)
    else:
        best_alpha = alpha

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        test_indices = dataset.get_split_indices(split=i, phase=phase)

        # Train features 및 scores 가져오기
        train_features = features[train_indices]
        train_scores = scores[train_indices]

        # 회귀 모델 학습
        regressor = Ridge(alpha=best_alpha).fit(train_features, train_scores)

        # Test features 및 scores 가져오기
        test_features = features[test_indices]
        test_scores = scores[test_indices]

        # 예측 수행
        preds = regressor.predict(test_features)
        preds = preds.flatten()

        # Debugging: 예측 값 및 실제 라벨 확인
        print(f"Predictions: {preds[:10]}")
        print(f"Test Scores: {test_scores.flatten()[:10]}")

        # SROCC 및 PLCC 계산
        srocc_value = stats.spearmanr(preds, test_scores.flatten())[0]
        plcc_value = stats.pearsonr(preds, test_scores.flatten())[0]
        print(f"SROCC: {srocc_value}, PLCC: {plcc_value}")

        srocc_dataset["global"].append(srocc_value)
        plcc_dataset["global"].append(plcc_value)

    return srocc_dataset, plcc_dataset, regressor, best_alpha, best_worst_results


def get_features_scores(model, dataloader, device, eval_type):
    scores = np.array([])  # 초기화
    mos = np.array([])  # 초기화

    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for i, batch in enumerate(dataloader):
            print(f"Batch {i} keys: {batch.keys()}")

            # Check if the expected keys are present in the batch
            if not all(key in batch for key in ['img_A_orig', 'img_B_orig']):
                print(f"Missing keys in batch {i}: {[key for key in ['img_A_orig', 'img_B_orig'] if key not in batch]}")
                continue

            print(f"Batch {i} mos: {batch['mos']}")  # Debugging: 'mos'의 내용 확인

            # Convert 'mos' to numpy array if it is a list
            if isinstance(batch['mos'], list):
                mos_batch = np.array(batch['mos'])
            else:
                mos_batch = batch['mos'].cpu().numpy()  # Ensure it is on CPU and convert to numpy

            mos = np.concatenate((mos, mos_batch), axis=0)  # Concatenate the mos

            # 이미지 데이터 가져오기
            img_A_orig = batch["img_A_orig"].to(device)
            img_B_orig = batch["img_B_orig"].to(device)

            # Check shapes
            print(f"img_A_orig shape: {img_A_orig.shape}, img_B_orig shape: {img_B_orig.shape}")  # Shape 확인

            # 모델에 대한 피처 추출
            with torch.amp.autocast(device_type='cuda'):
                feature_A, feature_B = model(img_A_orig, img_B_orig)  # 모델에서 두 개의 피처를 얻습니다.

def alpha_grid_search(dataset: Dataset,
                      features: np.ndarray,
                      scores: np.ndarray,
                      num_splits: int) -> float:


    grid_search_range = [1e-3, 1e3, 100]
    alphas = np.geomspace(*grid_search_range, endpoint=True)
    srocc_all = [[] for _ in range(len(alphas))]

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        val_indices = dataset.get_split_indices(split=i, phase="val")

        # for each index generate 5 indices (one for each crop)
        train_indices = np.repeat(train_indices * 5, 5) + np.tile(np.arange(5), len(train_indices))
        val_indices = np.repeat(val_indices * 5, 5) + np.tile(np.arange(5), len(val_indices))

        train_features = features[train_indices]
        train_scores = scores[train_indices]

        val_features = features[val_indices]
        val_scores = scores[val_indices]
        val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one

        for idx, alpha in enumerate(alphas):
            regressor = Ridge(alpha=alpha).fit(train_features, train_scores)
            preds = regressor.predict(val_features)
            preds = np.mean(np.reshape(preds, (-1, 5)), 1)  # Average the predictions of the 5 crops of the same image
            srocc_all[idx].append(stats.spearmanr(preds, val_scores)[0])

    srocc_all_median = [np.median(srocc) for srocc in srocc_all]
    srocc_all_median = np.array(srocc_all_median)
    best_alpha_idx = np.argmax(srocc_all_median)
    best_alpha = alphas[best_alpha_idx]

    return best_alpha


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    args, _ = parser.parse_known_args()
    config = parse_config(args.config)
    args = parse_command_line_args(config)
    args = merge_configs(config, args)

    args.data_base_path = Path(args.data_base_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로더 설정
    train_dataset = TID2013Dataset(Path('E:/ARNIQA/ARNIQA/dataset/TID2013'), phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)

    # Optimizer 및 모델 초기화
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.amp.GradScaler()

    # 훈련 시작
    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)
 """


## 지울거
""" 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import openpyxl
import pandas
from openpyxl.styles import Alignment
import pickle
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
import argparse
from tqdm import tqdm
from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR

synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "spaq"]

def save_checkpoint(model: nn.Module, checkpoint_path: Path, epoch: int, srocc: float) -> None:
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

def calculate_srcc_plcc(proj_A, proj_B):
    # 모델 출력값을 넘파이 배열로 변환
    proj_A = proj_A.detach().cpu().numpy()
    proj_B = proj_B.detach().cpu().numpy()

    # SRCC 계산
    srocc, _ = stats.spearmanr(proj_A.flatten(), proj_B.flatten())

    # PLCC 계산
    plcc, _ = stats.pearsonr(proj_A.flatten(), proj_B.flatten())

    return srocc, plcc



def train(args: DotMap,
          model: nn.Module,
          train_dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
          scaler: torch.cuda.amp.GradScaler,
          device: torch.device) -> None:

    checkpoint_path = Path(args.checkpoint_base_path) / args.experiment_name / "pretrain"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    print("Saving checkpoints in folder: ", checkpoint_path)

    start_epoch = 0
    max_epochs = args.training.epochs
    best_srocc = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch + 1}/{max_epochs}]")

        for i, batch in enumerate(progress_bar):
            inputs_A_orig = batch["img_A_orig"].to(device=device, non_blocking=True)
            inputs_A_ds = batch["img_A_ds"].to(device=device, non_blocking=True)

            inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=1)
            inputs_A = inputs_A.view(-1, 4, 3, 224, 224)

            inputs_B_orig = batch["img_B_orig"].to(device=device, non_blocking=True)
            inputs_B_ds = batch["img_B_ds"].to(device=device, non_blocking=True)

            inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=1)
            inputs_B = inputs_B.view(-1, 4, 3, 224, 224)

            # 디버깅: inputs_A와 inputs_B 값 확인
            print(f"[DEBUG] Epoch {epoch + 1}, Batch {i + 1}")
            print(f"inputs_A shape: {inputs_A.shape}, inputs_B shape: {inputs_B.shape}")
            print(f"inputs_A values (mean/std): {inputs_A.mean().item():.4f} / {inputs_A.std().item():.4f}")
            print(f"inputs_B values (mean/std): {inputs_B.mean().item():.4f} / {inputs_B.std().item():.4f}")

            optimizer.zero_grad()

            # Forward + backward + optimize
            with torch.amp.autocast(device_type='cuda'):
                proj_A, proj_B = model(inputs_A, inputs_B)

                # 디버깅: proj_A와 proj_B 값 확인
                print(f"proj_A shape: {proj_A.shape}, proj_B shape: {proj_B.shape}")
                print(f"proj_A values (mean/std): {proj_A.mean().item():.4f} / {proj_A.std().item():.4f}")
                print(f"proj_B values (mean/std): {proj_B.mean().item():.4f} / {proj_B.std().item():.4f}")

                loss = model.compute_loss(proj_A, proj_B)

            if torch.isnan(loss):
                raise ValueError("Loss is NaN")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cur_loss = loss.item()
            running_loss += cur_loss

            # 디버깅: 손실 값 확인
            print(f"Batch {i + 1} Loss: {cur_loss:.4f}")

            srocc, plcc = calculate_srcc_plcc(proj_A, proj_B)
            progress_bar.set_postfix(loss=running_loss / (i + 1), SRCC=srocc, PLCC=plcc)

        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, srocc)

    print('Finished training')

def validate(args: DotMap,
             model: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    model.eval()
    
    # KADID10K 데이터셋 및 SPAQ 데이터셋 사용
    datasets = ['kadid10k']
    for dataset_name in datasets:
        print(f"Validating dataset: {dataset_name}")

    srocc_all, plcc_all, _, _, _ = get_results(model=model, data_base_path=args.data_base_path,
                                               datasets=datasets,
                                               num_splits=args.validation.num_splits,
                                               phase="val", alpha=args.validation.alpha, grid_search=False,
                                               crop_size=args.test.crop_size, batch_size=args.test.batch_size,
                                               num_workers=args.test.num_workers, device=device)

    # Compute the median for each list in srocc_all and plcc_all
    srocc_all_median = {key: np.median(value["global"]) for key, value in srocc_all.items()}
    plcc_all_median = {key: np.median(value["global"]) for key, value in plcc_all.items()}

    # Compute the global average
    srocc_avg = np.mean(list(srocc_all_median.values()))
    plcc_avg = np.mean(list(plcc_all_median.values()))

    return srocc_avg, plcc_avg




def get_results(model: nn.Module,
                data_base_path: Path,
                datasets: List[str],
                num_splits: int,
                phase: str,
                alpha: float,
                grid_search: bool,
                crop_size: int,
                batch_size: int,
                num_workers: int,
                device: torch.device,
                eval_type: str = "scratch") -> Tuple[dict, dict, dict, dict, dict]:
    srocc_all = {}
    plcc_all = {}
    regressors = {}
    alphas = {}
    best_worst_results_all = {}

    assert phase in ["val", "test"], "Phase must be in ['val', 'test']"

    print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - Starting {phase} phase")
    for d in datasets:
        if d == "live":
            dataset = LIVEDataset(data_base_path / "LIVE", phase="all", crop_size=crop_size)
        elif d == "csiq":
            dataset = CSIQDataset(data_base_path / "CSIQ", phase="all", crop_size=crop_size)
        elif d == "tid2013":
            dataset = TID2013Dataset(data_base_path / "TID2013", phase="all", crop_size=crop_size)
        elif d == "kadid10k":
            dataset = KADID10KDataset(data_base_path / "KADID10K", phase="all", crop_size=crop_size)
        elif d == "flive":
            dataset = FLIVEDataset(data_base_path / "FLIVE", phase="all", crop_size=crop_size)
        elif d == "spaq":
            dataset = SPAQDataset(data_base_path / "SPAQ", phase="all", crop_size=crop_size)
        else:
            raise ValueError(f"Dataset {d} not supported")

        # 결과 계산
        srocc_dataset, plcc_dataset, regressor, alpha_value, best_worst_results = compute_metrics(model, dataset,
                                                                                                num_splits, phase,
                                                                                                alpha, grid_search,
                                                                                                batch_size, num_workers,
                                                                                                device, eval_type)
        srocc_all[d] = srocc_dataset
        plcc_all[d] = plcc_dataset
        regressors[d] = regressor
        alphas[d] = alpha_value
        best_worst_results_all[d] = best_worst_results
        print(f"{datetime.now().strftime('%d/%m/%Y %H:%M:%S')} - {d}:" f" SRCC: {np.median(srocc_dataset['global']):.3f} - PLCC: {np.median(plcc_dataset['global']):.3f}")

    return srocc_all, plcc_all, regressors, alphas, best_worst_results_all

def compute_metrics(model: nn.Module,
                    dataset: DataLoader,
                    num_splits: int,
                    phase: str,
                    alpha: float,
                    grid_search: bool,
                    batch_size: int,
                    num_workers: int,
                    device: torch.device,
                    eval_type: str = "scratch") -> Tuple[dict, dict, Ridge, float, dict]:
    srocc_dataset = {"global": []}
    plcc_dataset = {"global": []}
    best_worst_results = {}

    # DataLoader 설정
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # features 및 scores 가져오기
    features, scores = get_features_scores(model, dataloader, device, eval_type)

    # Debugging: features와 scores의 첫 10개 값을 확인
    print(f"Features: {features[:10]}")
    print(f"Scores: {scores[:10]}")

    # Grid search 또는 alpha 값을 사용하여 회귀 모델 학습
    if phase == "test" and grid_search:
        best_alpha = alpha_grid_search(dataset=dataset, features=features, scores=scores, num_splits=num_splits)
    else:
        best_alpha = alpha

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        test_indices = dataset.get_split_indices(split=i, phase=phase)

        # Train features 및 scores 가져오기
        train_features = features[train_indices]
        train_scores = scores[train_indices]

        # 회귀 모델 학습
        regressor = Ridge(alpha=best_alpha).fit(train_features, train_scores)

        # Test features 및 scores 가져오기
        test_features = features[test_indices]
        test_scores = scores[test_indices]

        # 예측 수행
        preds = regressor.predict(test_features)
        preds = preds.flatten()

        # Debugging: 예측 값 및 실제 라벨 확인
        print(f"Predictions: {preds[:10]}")
        print(f"Test Scores: {test_scores.flatten()[:10]}")

        # SROCC 및 PLCC 계산
        srocc_value = stats.spearmanr(preds, test_scores.flatten())[0]
        plcc_value = stats.pearsonr(preds, test_scores.flatten())[0]
        print(f"SROCC: {srocc_value}, PLCC: {plcc_value}")

        srocc_dataset["global"].append(srocc_value)
        plcc_dataset["global"].append(plcc_value)

    return srocc_dataset, plcc_dataset, regressor, best_alpha, best_worst_results


def get_features_scores(model, dataloader, device, eval_type):
    scores = np.array([])  # 초기화
    mos = np.array([])  # 초기화

    model.eval()  # 모델을 평가 모드로 설정
    with torch.no_grad():  # 그래디언트 계산 비활성화
        for i, batch in enumerate(dataloader):
            print(f"Batch {i} keys: {batch.keys()}")

            # Check if the expected keys are present in the batch
            if not all(key in batch for key in ['img_A_orig', 'img_B_orig']):
                print(f"Missing keys in batch {i}: {[key for key in ['img_A_orig', 'img_B_orig'] if key not in batch]}")
                continue

            print(f"Batch {i} mos: {batch['mos']}")  # Debugging: 'mos'의 내용 확인

            # Convert 'mos' to numpy array if it is a list
            if isinstance(batch['mos'], list):
                mos_batch = np.array(batch['mos'])
            else:
                mos_batch = batch['mos'].cpu().numpy()  # Ensure it is on CPU and convert to numpy

            mos = np.concatenate((mos, mos_batch), axis=0)  # Concatenate the mos

            # 이미지 데이터 가져오기
            img_A_orig = batch["img_A_orig"].to(device)
            img_B_orig = batch["img_B_orig"].to(device)

            # Check shapes
            print(f"img_A_orig shape: {img_A_orig.shape}, img_B_orig shape: {img_B_orig.shape}")  # Shape 확인

            # 모델에 대한 피처 추출
            with torch.amp.autocast(device_type='cuda'):
                feature_A, feature_B = model(img_A_orig, img_B_orig)  # 모델에서 두 개의 피처를 얻습니다.

def alpha_grid_search(dataset: Dataset,
                      features: np.ndarray,
                      scores: np.ndarray,
                      num_splits: int) -> float:

    grid_search_range = [1e-3, 1e3, 100]
    alphas = np.geomspace(*grid_search_range, endpoint=True)
    srocc_all = [[] for _ in range(len(alphas))]

    for i in range(num_splits):
        train_indices = dataset.get_split_indices(split=i, phase="train")
        val_indices = dataset.get_split_indices(split=i, phase="val")

        # for each index generate 5 indices (one for each crop)
        train_indices = np.repeat(train_indices * 5, 5) + np.tile(np.arange(5), len(train_indices))
        val_indices = np.repeat(val_indices * 5, 5) + np.tile(np.arange(5), len(val_indices))

        train_features = features[train_indices]
        train_scores = scores[train_indices]

        val_features = features[val_indices]
        val_scores = scores[val_indices]
        val_scores = val_scores[::5]  # Scores are repeated for each crop, so we only keep the first one

        for idx, alpha in enumerate(alphas):
            regressor = Ridge(alpha=alpha).fit(train_features, train_scores)
            preds = regressor.predict(val_features)
            preds = np.mean(np.reshape(preds, (-1, 5)), 1)  # Average the predictions of the 5 crops of the same image
            srocc_all[idx].append(stats.spearmanr(preds, val_scores)[0])

    srocc_all_median = [np.median(srocc) for srocc in srocc_all]
    srocc_all_median = np.array(srocc_all_median)
    best_alpha_idx = np.argmax(srocc_all_median)
    best_alpha = alphas[best_alpha_idx]

    return best_alpha



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    args, _ = parser.parse_known_args()
    config = parse_config(args.config)
    args = parse_command_line_args(config)
    args = merge_configs(config, args)

    args.data_base_path = Path(args.data_base_path)  # 경로를 Path 객체로 변환
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 로더 설정
    train_dataset = KADID10KDataset(Path('E:/ARNIQA - SE/ARNIQA/dataset/KADID10K'), phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)

    # SPAQDataset 로드
    #spaq_dataset = SPAQDataset(root='E:/ARNIQA/ARNIQA/dataset/SPAQ', phase='train')
    #print(f"Loaded {len(spaq_dataset)} images from SPAQDataset.")

    # Optimizer 및 모델 초기화
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.amp.GradScaler()

    # 훈련 시작
    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)
 """