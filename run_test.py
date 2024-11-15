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
from typing import List, Tuple
from datetime import datetime
from einops import rearrange
from sklearn.linear_model import Ridge
from scipy import stats
import argparse

from data import LIVEDataset, CSIQDataset, TID2013Dataset, KADID10KDataset, FLIVEDataset, SPAQDataset
from utils.utils import PROJECT_ROOT, parse_command_line_args, merge_configs, parse_config
from models.simclr import SimCLR

synthetic_datasets = ["live", "csiq", "tid2013", "kadid10k"]
authentic_datasets = ["flive", "spaq"]

# 전역 범위로 collate_fn 함수 정의
def custom_collate_fn(batch):
    # None 값 필터링
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return {
            "img_A_orig": torch.empty(0),
            "img_B_orig": torch.empty(0),
            "img_A_ds": torch.empty(0),
            "img_B_ds": torch.empty(0),
            "mos": torch.empty(0),
            "distortion_type": []
        }
    return torch.utils.data.dataloader.default_collate(batch)

def test(args: DotMap, model: nn.Module, device: torch.device) -> None:
    checkpoint_base_path = PROJECT_ROOT / "experiments"
    checkpoint_path = checkpoint_base_path / args.experiment_name
    regressor_path = checkpoint_path / "regressors"
    regressor_path.mkdir(parents=True, exist_ok=True)

    eval_type = args.get("eval_type", "scratch")
    model.eval()

    srocc_all, plcc_all, regressors, alphas, best_worst_results_all = get_results(
        model=model,
        data_base_path=args.data_base_path,
        datasets=args.test.datasets,
        num_splits=args.test.num_splits,
        phase="test",
        alpha=args.test.alpha,
        grid_search=args.test.grid_search,
        crop_size=args.test.crop_size,
        batch_size=args.test.batch_size,
        num_workers=args.test.num_workers,
        device=device,
        eval_type=eval_type
    )

    # Check if all datasets are in the results and handle missing datasets
    srocc_all_median = {key: np.median(value["global"]) for key, value in srocc_all.items()}
    plcc_all_median = {key: np.median(value["global"]) for key, value in plcc_all.items()}

    # Make sure all synthetic datasets are included in srocc_all_median, handle missing ones
    for dataset in synthetic_datasets:
        if dataset not in srocc_all_median:
            srocc_all_median[dataset] = float('nan')  # handle missing
        if dataset not in plcc_all_median:
            plcc_all_median[dataset] = float('nan')  # handle missing

    for dataset in authentic_datasets:
        if dataset not in srocc_all_median:
            srocc_all_median[dataset] = float('nan')  # handle missing
        if dataset not in plcc_all_median:
            plcc_all_median[dataset] = float('nan')  # handle missing

    srocc_synthetic_avg = np.nanmean([srocc_all_median[key] for key in synthetic_datasets])
    plcc_synthetic_avg = np.nanmean([plcc_all_median[key] for key in synthetic_datasets])
    srocc_authentic_avg = np.nanmean([srocc_all_median[key] for key in authentic_datasets])
    plcc_authentic_avg = np.nanmean([plcc_all_median[key] for key in authentic_datasets])
    srocc_avg = np.nanmean(list(srocc_all_median.values()))
    plcc_avg = np.nanmean(list(plcc_all_median.values()))

    print(f"{'Dataset':<15} {'Alpha':<15} {'SROCC':<15} {'PLCC':<15}")
    for dataset in srocc_all_median.keys():
        print(f"{dataset:<15} {alphas.get(dataset, 'N/A')} {srocc_all_median[dataset]:<15.4f} {plcc_all_median.get(dataset, 'N/A'):<15.4f}")
    print(f"{'Synthetic avg':<15} {srocc_synthetic_avg:<15.4f} {plcc_synthetic_avg:<15.4f}")
    print(f"{'Authentic avg':<15} {srocc_authentic_avg:<15.4f} {plcc_authentic_avg:<15.4f}")
    print(f"{'Global avg':<15} {srocc_avg:<15.4f} {plcc_avg:<15.4f}")

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

        srocc_dataset, plcc_dataset, regressor, alpha, best_worst_results = compute_metrics(
            model, dataset, num_splits, phase, alpha, grid_search, batch_size, num_workers, device, eval_type)
        
        srocc_all[d] = srocc_dataset
        plcc_all[d] = plcc_dataset
        regressors[d] = regressor
        alphas[d] = alpha
        best_worst_results_all[d] = best_worst_results

    return srocc_all, plcc_all, regressors, alphas, best_worst_results_all

def compute_metrics(model: nn.Module,
                    dataset: Dataset,
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

    # DataLoader에서 전역 범위의 custom_collate_fn을 사용
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=custom_collate_fn)

    features, scores = get_features_scores(model, dataloader, device, eval_type)

    # Ridge regression to calculate SROCC and PLCC
    regressor = Ridge(alpha=alpha).fit(features, scores)
    preds = regressor.predict(features)

    srocc_dataset["global"].append(stats.spearmanr(preds, scores)[0])
    plcc_dataset["global"].append(stats.pearsonr(preds, scores)[0])

    return srocc_dataset, plcc_dataset, regressor, alpha, best_worst_results

def get_features_scores(model: nn.Module,
                        dataloader: DataLoader,
                        device: torch.device,
                        eval_type: str = "scratch") -> Tuple[np.ndarray, np.ndarray]:
    feats = []
    scores = []

    for batch in dataloader:
        img_A_orig = batch["img_A_orig"].to(device).squeeze(1).squeeze(1)  # [batch_size, 3, H, W]
        img_A_ds = batch["img_A_ds"].to(device).squeeze(1).squeeze(1)      # [batch_size, 3, H, W]
        mos = batch["mos"]

        # 필요한 형식으로 차원 조정: [batch_size, 4, 3, H, W]로 맞춤
        img_A_orig = img_A_orig.unsqueeze(1).expand(-1, 4, -1, -1, -1)  # [batch_size, num_crops, 3, H, W]
        img_A_ds = img_A_ds.unsqueeze(1).expand(-1, 4, -1, -1, -1)      # [batch_size, num_crops, 3, H, W]

        print("Before model pass - img_A shape:", img_A_orig.shape, ", img_B shape:", img_A_ds.shape)

        # 모델에 img_A_orig과 img_A_ds를 입력
        with torch.cuda.amp.autocast(), torch.no_grad():
            f_orig, f_ds = model(img_A_orig, img_A_ds)
            f = torch.hstack((f_orig, f_ds))

        print("After model pass - proj_A shape:", f_orig.shape, ", proj_B shape:", f_ds.shape)

        num_crops = img_A_orig.shape[1]
        repeated_mos = mos.repeat_interleave(num_crops)

        feats.append(f.cpu().numpy())
        scores.append(repeated_mos.cpu().numpy())

    return np.vstack(feats), np.hstack(scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Configuration file')
    parser.add_argument("--eval_type", type=str, default="scratch", choices=["scratch", "arniqa"],
                        help="Whether to test a model trained from scratch or the one pretrained by the authors of the"
                             "paper. Must be in ['scratch', 'arniqa']")
    args, _ = parser.parse_known_args()
    eval_type = args.eval_type
    config = parse_config(args.config)
    args = parse_command_line_args(config)
    args = merge_configs(config, args)
    args.eval_type = eval_type
    args.data_base_path = Path(args.data_base_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.eval_type == "scratch":
        model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
        checkpoint_base_path = PROJECT_ROOT / "experiments"
        assert (checkpoint_base_path / args.experiment_name).exists(), \
            f"Experiment {(checkpoint_base_path / args.experiment_name)} does not exist"
        checkpoint_path = Path("E:/ARNIQA/ARNIQA/experiments/my_experiment/pretrain/epoch_9_srocc_0.960.pth")

        # 체크포인트를 불러오기
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=True)

    elif args.eval_type == "arniqa":
        model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA")
    else:
        raise ValueError(f"Eval type {args.eval_type} not supported")
    model.to(device)
    model.eval()

    test(args, model, device)
