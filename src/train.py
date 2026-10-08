import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from pathlib import Path
import numpy as np
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from prepare_dataset import CoberturaDataset


class DiceLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = torch.zeros_like(probs)
        mask = targets != self.ignore_index
        targets_clamped = targets.clone()
        targets_clamped[~mask] = 0
        one_hot.scatter_(1, targets_clamped.unsqueeze(1), 1)
        one_hot = one_hot * mask.unsqueeze(1).float()

        intersection = (probs * one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, num_classes, ignore_index=0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(num_classes, ignore_index)

    def forward(self, logits, targets):
        return self.ce(logits, targets) + self.dice(logits, targets)


def calcular_iou(pred, target, num_classes, ignore_index=0):
    ious = []
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_mask = pred == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return np.mean(ious) if ious else 0.0


def entrenar_epoca(model, dataloader, optimizer, criterion, device, num_classes):
    model.train()
    total_loss = 0
    total_iou = 0
    n_batches = 0

    pbar = tqdm(dataloader, desc="Entrenando")
    for imagenes, etiquetas in pbar:
        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=imagenes)
        logits = outputs.logits
        logits = torch.nn.functional.interpolate(
            logits, size=etiquetas.shape[1:], mode="bilinear", align_corners=False
        )

        loss = criterion(logits, etiquetas)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        iou = calcular_iou(pred, etiquetas, num_classes)
        total_iou += iou
        n_batches += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{iou:.4f}"})

    return total_loss / n_batches, total_iou / n_batches


def validar(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    total_iou = 0
    n_batches = 0

    with torch.no_grad():
        for imagenes, etiquetas in tqdm(dataloader, desc="Validando"):
            imagenes = imagenes.to(device)
            etiquetas = etiquetas.to(device)

            outputs = model(pixel_values=imagenes)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits, size=etiquetas.shape[1:], mode="bilinear", align_corners=False
            )

            loss = criterion(logits, etiquetas)
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            iou = calcular_iou(pred, etiquetas, num_classes)
            total_iou += iou
            n_batches += 1

    return total_loss / n_batches, total_iou / n_batches


def main():
    root = Path(__file__).parent.parent
    with open(root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    num_classes = config["etiquetas"]["num_clases"]
    model_name = config["modelo"]["nombre"]
    batch_size = config["modelo"]["batch_size"]
    epochs = config["modelo"]["epochs"]
    lr = config["modelo"]["lr"]
    weight_decay = config["modelo"]["weight_decay"]
    patience = config["modelo"]["early_stopping_patience"]
    num_workers = config["entrenamiento"]["num_workers"]

    print(f"Cargando modelo: {model_name}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        reshape_last_stage=True
    )
    model.to(device)

    patches_dir = Path(config["paths"]["patches"])
    train_idx = np.load(patches_dir / "train_indices.npy", allow_pickle=True)
    val_idx = np.load(patches_dir / "val_indices.npy", allow_pickle=True)

    train_dataset = CoberturaDataset(str(patches_dir), train_idx.tolist())
    val_dataset = CoberturaDataset(str(patches_dir), val_idx.tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = CombinedLoss(num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    epochs_no_mejor = 0
    models_dir = Path(config["paths"]["models"])
    models_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_iou = entrenar_epoca(model, train_loader, optimizer, criterion, device, num_classes)
        val_loss, val_iou = validar(model, val_loader, criterion, device, num_classes)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_mejor = 0
            torch.save(model.state_dict(), models_dir / "best_model.pth")
            print("Modelo guardado (mejor val_loss)")
        else:
            epochs_no_mejor += 1
            if epochs_no_mejor >= patience:
                print(f"Early stopping despues de {epoch+1} epocas")
                break

    print("\nEntrenamiento completado")
    print(f"Mejor modelo guardado en: {models_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
