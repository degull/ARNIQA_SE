import torch
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from data import KADID10KDataset
from models.simclr import SimCLR
from scipy.stats import spearmanr, pearsonr
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from dotmap import DotMap
import torch.nn as nn

# SimCLR 모델 로드 함수
def load_simclr_model(model_path, device, encoder_params, temperature):
    encoder_params = DotMap(encoder_params)
    model = SimCLR(encoder_params=encoder_params, temperature=temperature).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Ridge Regressor 로드 함수
def load_ridge_regressor(regressor_path):
    with open(regressor_path, "rb") as f:
        regressor = pickle.load(f)
    print("[INFO] Ridge Regressor loaded.")
    return regressor

# Ridge Regressor 평가
def evaluate_ridge_regressor(regressor, model: nn.Module, val_dataloader: DataLoader, device: torch.device):
    model.eval()
    mos_scores, predictions = [], []

    with torch.no_grad():
        for batch in val_dataloader:
            inputs_A = batch["img_A"].to(device)
            mos = batch["mos"]

            if inputs_A.dim() == 4:
                inputs_A = inputs_A.unsqueeze(1)
                inputs_A = inputs_A.expand(-1, 2, -1, -1, -1)

            proj_A, _ = model(inputs_A, inputs_A)
            prediction = regressor.predict(proj_A.cpu().numpy())

            # MOS 점수 크기 맞추기
            if proj_A.shape[0] > mos.shape[0]:
                mos_repeated = np.repeat(mos.numpy(), proj_A.shape[0] // mos.shape[0])[:proj_A.shape[0]]
            else:
                mos_repeated = mos.numpy()[:proj_A.shape[0]]

            predictions.append(prediction)
            mos_scores.append(mos_repeated)

    mos_scores = np.hstack(mos_scores)
    predictions = np.hstack(predictions)

    # Debug: 크기 확인
    print(f"Final mos_scores shape: {mos_scores.shape}, predictions shape: {predictions.shape}")
    return mos_scores, predictions

# 결과 그래프 출력
def plot_results(mos_scores, predictions):
    print("[INFO] Plotting results...")
    assert mos_scores.shape == predictions.shape, "MOS scores and predictions must have the same shape."
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
    # Paths
    model_path = "E:/ARNIQA - SE/ARNIQA/experiments/my_experiment/regressors/simclr_model.pth"
    regressor_path = "E:/ARNIQA - SE/ARNIQA/experiments/my_experiment/regressors/ridge_regressor.pkl"
    dataset_path = "E:/ARNIQA - SE/ARNIQA/dataset/KADID10K/kadid10k.csv"

    # Configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_params = {"embedding_dim": 128, "pretrained": True, "use_norm": True}  # Update as per model
    temperature = 0.1

    # Load dataset
    dataset = KADID10KDataset(dataset_path)
    _, val_dataset, _ = random_split(dataset, [int(0.7 * len(dataset)), int(0.1 * len(dataset)), len(dataset) - int(0.7 * len(dataset)) - int(0.1 * len(dataset))])
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Load model and regressor
    model = load_simclr_model(model_path, device, encoder_params, temperature)
    regressor = load_ridge_regressor(regressor_path)

    # Evaluate and plot
    mos_scores, predictions = evaluate_ridge_regressor(regressor, model, val_dataloader, device)
    val_srcc, _ = spearmanr(mos_scores, predictions)
    val_plcc, _ = pearsonr(mos_scores, predictions)

    plot_results(mos_scores, predictions)
