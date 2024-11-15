import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from dotmap import DotMap
import openpyxl
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
    """모델 체크포인트 저장 함수."""
    filename = f"epoch_{epoch}_srocc_{srocc:.3f}.pth"
    torch.save(model.state_dict(), checkpoint_path / filename)

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
            inputs_A_orig = batch["img_A_orig"].to(device=device, non_blocking=True)  # Shape: [1, num_crops, 3, 224, 224]
            inputs_A_ds = batch["img_A_ds"].to(device=device, non_blocking=True)

            # Concatenate along the batch dimension
            inputs_A = torch.cat((inputs_A_orig, inputs_A_ds), dim=0)

            inputs_B_orig = batch["img_B_orig"].to(device=device, non_blocking=True)
            inputs_B_ds = batch["img_B_ds"].to(device=device, non_blocking=True)

            inputs_B = torch.cat((inputs_B_orig, inputs_B_ds), dim=0)

            # Print shapes to debug
            print(f"inputs_A shape: {inputs_A.shape}, inputs_B shape: {inputs_B.shape}")  # 디버깅을 위한 출력

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            with torch.cuda.amp.autocast():
                # Check before view
                print(f"Before view - inputs_A shape: {inputs_A.shape}, inputs_B shape: {inputs_B.shape}")

                inputs_A = inputs_A.view(-1, 3, 224, 224)
                inputs_B = inputs_B.view(-1, 3, 224, 224)

                # After reshaping
                print(f"After view - inputs_A shape: {inputs_A.shape}, inputs_B shape: {inputs_B.shape}")

                proj_A, proj_B = model(inputs_A, inputs_B)  # 모델의 출력 받기
                loss = model.compute_loss(proj_A, proj_B)  # 손실 계산

            if loss is None:
                print("Loss is None, check the forward method.")
            elif torch.isnan(loss):
                raise ValueError("Loss is NaN")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            cur_loss = loss.item()
            running_loss += cur_loss
            progress_bar.set_postfix(loss=running_loss / (i + 1), SROCC=best_srocc)

        # Validation
        if epoch % args.validation.frequency == 0:
            print("Starting validation...")
            last_srocc, last_plcc = validate(args, model, device)

        # Save checkpoints
        if epoch % args.checkpoint_frequency == 0:
            save_checkpoint(model, checkpoint_path, epoch, last_srocc)

    print('Finished training')

def validate(args: DotMap,
             model: nn.Module,
             device: torch.device) -> Tuple[float, float]:
    model.eval()
    
    # KADID10K 데이터셋 및 SPAQ 데이터셋 사용
    datasets = ['kadid10k', 'spaq']
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

    # 배치 형태 출력
    for i, batch in enumerate(dataloader):
        print(f"Batch {i} shapes:")
        print(f"img_A_orig shape: {batch['img_A_orig'].shape}")
        print(f"img_B_orig shape: {batch['img_B_orig'].shape}")
        print(f"mos shape: {batch['mos'].shape}")
        print("-" * 40)

    # Debugging: scores shape 확인
    print(f"Features shape: {features.shape}, Scores shape: {scores.shape}")

    # Grid search 또는 alpha 값을 사용하여 회귀 모델 학습
    if phase == "test" and grid_search:
        best_alpha = alpha_grid_search(dataset=dataset, features=features, scores=scores, num_splits=num_splits)
    else:
        best_alpha = alpha

    for i in range(num_splits):
        # Split 데이터셋
        train_indices = dataset.get_split_indices(split=i, phase="train")
        test_indices = dataset.get_split_indices(split=i, phase=phase)

        # Train features 및 scores 가져오기
        train_features = features[train_indices]
        train_scores = scores[train_indices]

        # Reshape to 2D if necessary
        if train_features.ndim == 1:
            train_features = train_features.reshape(-1, 1)
        if train_scores.ndim == 1:
            train_scores = train_scores.reshape(-1, 1)

        regressor = Ridge(alpha=best_alpha).fit(train_features, train_scores)

        # Test features 및 scores 가져오기
        test_features = features[test_indices]
        test_scores = scores[test_indices]

        # Reshape to 2D if necessary
        if test_features.ndim == 1:
            test_features = test_features.reshape(-1, 1)
        if test_scores.ndim == 1:
            test_scores = test_scores.reshape(-1, 1)

        preds = regressor.predict(test_features)

        # Reshape preds to 1D
        preds = preds.flatten()  # Make preds a 1D array

        # Debugging: preds와 test_scores의 shape 확인
        print(f"Preds shape: {preds.shape}, Test scores shape: {test_scores.shape}")

        # Check if preds needs to be reshaped to match test_scores
        if preds.shape[0] != test_scores.shape[0]:
            print(f"Mismatch in shapes: preds {preds.shape}, test_scores {test_scores.shape}")
            preds = preds[:test_scores.shape[0]]  # Adjust preds to match test_scores if necessary

        # Compute SROCC 및 PLCC
        srocc_dataset["global"].append(stats.spearmanr(preds, test_scores.flatten())[0])  # Flatten for correlation
        plcc_dataset["global"].append(stats.pearsonr(preds, test_scores.flatten())[0])  # Flatten for correlation

        # Compute best and worst results
        if i == 0:
            diff = np.abs(preds - test_scores.flatten())
            sorted_diff_indices = np.argsort(diff)
            best_indices = sorted_diff_indices[:16]
            worst_indices = sorted_diff_indices[-16:][::-1]
            best_worst_results["best"] = {"images": dataset.images[test_indices[best_indices]], "gts": test_scores[best_indices], "preds": preds[best_indices]}
            best_worst_results["worst"] = {"images": dataset.images[test_indices[worst_indices]], "gts": test_scores[worst_indices], "preds": preds[worst_indices]}

    return srocc_dataset, plcc_dataset, regressor, best_alpha, best_worst_results

def calculate_score(feature_A, feature_B):
    """
    Calculate similarity score based on two feature sets.
    
    Args:
        feature_A (torch.Tensor): Feature tensor for image A.
        feature_B (torch.Tensor): Feature tensor for image B.
    
    Returns:
        torch.Tensor: Calculated similarity score.
    """
    # Check if features are tuples and extract the actual tensor if necessary
    if isinstance(feature_A, tuple):
        feature_A = feature_A[0]  # Assuming the first element is the feature tensor
    if isinstance(feature_B, tuple):
        feature_B = feature_B[0]  # Assuming the first element is the feature tensor

    # L2 정규화
    feature_A_norm = feature_A / feature_A.norm(dim=1, keepdim=True)
    feature_B_norm = feature_B / feature_B.norm(dim=1, keepdim=True)

    # 코사인 유사성 계산
    score = (feature_A_norm * feature_B_norm).sum(dim=1)  # 배치의 각 피처에 대해 유사성 계산

    return score

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
    """
    Perform a grid search over the validation splits to find the best alpha value for the regression based on the SROCC
    metric. The grid search is performed over the range [1-e3, 1e3, 100].

    Args:
        dataset (Dataset): dataset to use
        features (np.ndarray): features extracted with the model to test
        scores (np.ndarray): ground-truth MOS scores
        num_splits (int): number of splits to use

    Returns:
        alpha (float): best alpha value
    """

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
    train_dataset = KADID10KDataset(args.data_base_path / "KADID10K", phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, shuffle=True, num_workers=args.training.num_workers)

    # SPAQDataset 로드
    spaq_dataset = SPAQDataset(root='E:/ARNIQA/ARNIQA/dataset/SPAQ', phase='train')
    print(f"Loaded {len(spaq_dataset)} images from SPAQDataset.")

    # Optimizer 및 모델 초기화
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.training.step_size, gamma=args.training.gamma)
    scaler = torch.amp.GradScaler()

    # 훈련 시작
    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)