import yaml
import numpy as np
import rasterio
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch


class CoberturaDataset(Dataset):
    def __init__(self, patches_dir, indices_list, transform=None):
        self.patches_dir = Path(patches_dir)
        self.indices_list = indices_list
        self.transform = transform

    def __len__(self):
        return len(self.indices_list)

    def __getitem__(self, idx):
        nombre = self.indices_list[idx]
        ruta_img = self.patches_dir / f"patch_{nombre}_img.npy"
        ruta_lbl = self.patches_dir / f"patch_{nombre}_lbl.npy"

        imagen = np.load(ruta_img)
        etiqueta = np.load(ruta_lbl)

        imagen = torch.from_numpy(imagen).float()
        etiqueta = torch.from_numpy(etiqueta).long()

        if self.transform:
            imagen = self.transform(imagen)

        return imagen, etiqueta


def crear_patches(ruta_imagen, ruta_etiquetas, output_dir, patch_size=256, overlap=0.5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ruta_imagen) as src_img:
        imagen = src_img.read()
        perfil_img = src_img.profile
        height = src_img.height
        width = src_img.width

    with rasterio.open(ruta_etiquetas) as src_lbl:
        etiquetas = src_lbl.read(1)

    stride = int(patch_size * (1 - overlap))
    parches = []

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch_img = imagen[:, y:y+patch_size, x:x+patch_size]
            patch_lbl = etiquetas[y:y+patch_size, x:x+patch_size]

            if patch_img.shape[1] == patch_size and patch_img.shape[2] == patch_size:
                if np.sum(patch_lbl > 0) > (patch_size * patch_size * 0.1):
                    idx = len(parches)
                    np.save(output_dir / f"patch_{idx:05d}_img.npy", patch_img)
                    np.save(output_dir / f"patch_{idx:05d}_lbl.npy", patch_lbl)
                    parches.append(f"{idx:05d}")

    print(f"Parches creados: {len(parches)}")
    return parches


def dividir_dataset(parches, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    train, temp = train_test_split(parches, train_size=train_ratio, random_state=seed)
    val, test = train_test_split(temp, train_size=val_ratio/(val_ratio+test_ratio), random_state=seed)
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    return train, val, test


def main():
    root = Path(__file__).parent.parent
    with open(root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ruta_imagen = Path(config["paths"]["raw"]) / "sentinel2_armenia_2024.tif"
    ruta_etiquetas = Path(config["paths"]["labels"]) / "etiquetas_armenia.tif"
    output_dir = Path(config["paths"]["patches"])

    patch_size = config["entrenamiento"]["patch_size"]
    overlap = config["entrenamiento"]["overlap"]

    parches = crear_patches(str(ruta_imagen), str(ruta_etiquetas), str(output_dir), patch_size, overlap)

    train, val, test = dividir_dataset(
        parches,
        config["entrenamiento"]["train_split"],
        config["entrenamiento"]["val_split"],
        config["entrenamiento"]["test_split"],
        config["project"]["seed"]
    )

    np.save(output_dir / "train_indices.npy", np.array(train))
    np.save(output_dir / "val_indices.npy", np.array(val))
    np.save(output_dir / "test_indices.npy", np.array(test))


if __name__ == "__main__":
    main()
