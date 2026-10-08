import yaml
import torch
import numpy as np
import rasterio
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
from tqdm import tqdm


def cargar_modelo(checkpoint_path, num_classes, device):
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b2-finetuned-ade-512-512",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        reshape_last_stage=True
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predecir_imagen(model, ruta_imagen, device, patch_size=256, batch_size=8):
    with rasterio.open(ruta_imagen) as src:
        imagen = src.read()
        perfil = src.profile
        height = src.height
        width = src.width
        n_bandas = src.count

    print(f"Imagen: {width}x{height}, {n_bandas} bandas")

    padding_h = (patch_size - height % patch_size) % patch_size
    padding_w = (patch_size - width % patch_size) % patch_size
    imagen_padded = np.pad(
        imagen,
        ((0, 0), (0, padding_h), (0, padding_w)),
        mode="reflect"
    )

    h_padded = imagen_padded.shape[1]
    w_padded = imagen_padded.shape[2]

    prediccion = np.zeros((h_padded, w_padded), dtype=np.uint8)

    parches = []
    posiciones = []

    for y in range(0, h_padded - patch_size + 1, patch_size):
        for x in range(0, w_padded - patch_size + 1, patch_size):
            patch = imagen_padded[:, y:y+patch_size, x:x+patch_size]
            parches.append(patch)
            posiciones.append((y, x))

    with torch.no_grad():
        for i in tqdm(range(0, len(parches), batch_size), desc="Prediciendo"):
            batch = parches[i:i+batch_size]
            batch_tensor = torch.from_numpy(np.stack(batch)).float().to(device)

            outputs = model(pixel_values=batch_tensor)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits, size=(patch_size, patch_size), mode="bilinear", align_corners=False
            )
            pred = logits.argmax(dim=1).cpu().numpy()

            for j, (y, x) in enumerate(posiciones[i:i+batch_size]):
                prediccion[y:y+patch_size, x:x+patch_size] = pred[j]

    prediccion = prediccion[:height, :width]

    return prediccion, perfil


def guardar_prediccion(prediccion, perfil, ruta_salida):
    perfil.update(
        dtype="uint8",
        count=1,
        compress="lzw",
        nodata=0
    )

    with rasterio.open(ruta_salida, "w", **perfil) as dst:
        dst.write(prediccion, 1)

    print(f"Prediccion guardada: {ruta_salida}")


def main():
    root = Path(__file__).parent.parent
    with open(root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    num_classes = config["etiquetas"]["num_clases"]
    checkpoint = Path(config["paths"]["models"]) / "best_model.pth"
    ruta_imagen = Path(config["paths"]["raw"]) / "sentinel2_armenia_2024.tif"
    ruta_salida = Path(config["paths"]["output"]) / "prediccion_armenia.tif"

    if not checkpoint.exists():
        print(f"No se encontro el modelo: {checkpoint}")
        print("Primero ejecute train.py para entrenar el modelo")
        return

    if not ruta_imagen.exists():
        print(f"No se encontro la imagen: {ruta_imagen}")
        return

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    model = cargar_modelo(str(checkpoint), num_classes, device)
    prediccion, perfil = predecir_imagen(model, str(ruta_imagen), device)
    guardar_prediccion(prediccion, perfil, str(ruta_salida))


if __name__ == "__main__":
    main()
