import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import random
import os
import numpy as np

from train import train  # train 함수는 이미 수정한 train.py에서 호출됨
from run_test import test  # test 함수는 모델 평가에 사용
from models.simclr import SimCLR  # SimCLR 모델 정의
from data import KADID10KDataset  # 데이터셋 정의
from utils.utils import PROJECT_ROOT, parse_config, parse_command_line_args, merge_configs  # 유틸리티 함수들

def main():
    # 명령줄 인수 및 설정 파일 불러오기
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="e:/ARNIQA/ARNIQA/config.yaml", help='Path to the configuration file')
    args, unknown = parser.parse_known_args()
    
    # 설정 파일을 로드하고 명령줄 인수를 병합
    config = parse_config(args.config)
    args = parse_command_line_args(config)
    args = merge_configs(config, args)  # 설정 파일과 명령줄 인수 병합
    print(args)

    # GPU 또는 CPU 설정
    if args.device != -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = torch.device("cpu")

    # 랜덤 시드 설정
    SEED = args.seed
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    np.random.seed(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 경로 설정
    args.data_base_path = Path(args.data_base_path)
    args.checkpoint_base_path = PROJECT_ROOT / "experiments"

    # 학습 데이터셋 및 데이터로더 초기화
    train_dataset = KADID10KDataset(root=args.data_base_path / "KADID10K", phase="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.training.batch_size, num_workers=args.training.num_workers,
                                  shuffle=True, pin_memory=True, drop_last=True)

    # 검증 데이터셋 및 데이터로더 초기화
    validation_dataset = KADID10KDataset(root=args.data_base_path / "KADID10K", phase="val")
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.test.batch_size, num_workers=args.test.num_workers,
                                       shuffle=False, pin_memory=True)

    # SimCLR 모델 초기화
    model = SimCLR(encoder_params=args.model.encoder, temperature=args.model.temperature)
    model = model.to(device)

    # Optimizer 초기화
    if args.training.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.training.lr,
                                     weight_decay=args.training.optimizer.weight_decay,
                                     betas=args.training.optimizer.betas, eps=args.training.optimizer.eps)
    elif args.training.optimizer.name == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.training.lr,
                                      weight_decay=args.training.optimizer.weight_decay,
                                      betas=args.training.optimizer.betas, eps=args.training.optimizer.eps)
    elif args.training.optimizer.name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.training.lr, momentum=args.training.optimizer.momentum,
                                    weight_decay=args.training.optimizer.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.training.optimizer.name} not implemented")

    # Scheduler 초기화
    if "lr_scheduler" in args.training and args.training.lr_scheduler.name == "CosineAnnealingWarmRestarts":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                            T_0=args.training.lr_scheduler.T_0,
                                                                            T_mult=args.training.lr_scheduler.T_mult,
                                                                            eta_min=args.training.lr_scheduler.eta_min,
                                                                            verbose=False)
    else:
        lr_scheduler = None

    # Mixed Precision Scaler 초기화
    scaler = torch.cuda.amp.GradScaler()

    # 훈련 재개 또는 새 훈련
    if args.training.resume_training:
        try:
            checkpoint_path = args.checkpoint_base_path / args.experiment_name / "pretrain"
            checkpoint_path = [el for el in checkpoint_path.glob("*.pth") if "last" in el.name][0]
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            epoch = checkpoint["epoch"]
            args.training.start_epoch = epoch + 1
            run_id = checkpoint["config"]["logging"]["wandb"].get("run_id", None)
            args.best_srocc = checkpoint["config"]["best_srocc"]
            print(f"--- Resuming training after epoch {epoch + 1} ---")
        except Exception:
            print("ERROR: Could not resume training. Starting from scratch.")

    # 학습 진행
    train(args, model, train_dataloader, optimizer, lr_scheduler, scaler, device)
    print("--- Training finished ---")

    # 최종 체크포인트를 불러와 테스트 시작
    checkpoint_path = args.checkpoint_base_path / args.experiment_name / "pretrain"
    checkpoint_path = [ckpt_path for ckpt_path in checkpoint_path.glob("*.pth") if "best" in ckpt_path.name][0]
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    print(f"Starting testing with best checkpoint...")

    # 테스트 진행
    test(args, model, None, device)
    print("--- Testing finished ---")


if __name__ == '__main__':
    main()
