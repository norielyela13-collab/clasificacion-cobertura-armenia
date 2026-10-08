import yaml
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


COLORES_IDEAM = {
    0: ("#000000", "No clasificado"),
    1: ("#006400", "Bosque denso"),
    2: ("#228B22", "Bosque fragmentado"),
    3: ("#32CD32", "Bosque de galeria"),
    4: ("#7CFC00", "Bosque secundario"),
    5: ("#ADFF2F", "Pastos"),
    6: ("#FFD700", "Agricultura"),
    7: ("#FF0000", "Zonas urbanas"),
    8: ("#0000FF", "Cuerpos de agua"),
    9: ("#8B4513", "Zonas desnudas"),
    10: ("#FFFFFF", "Nubes"),
    11: ("#9370DB", "Paramo"),
    12: ("#008080", "Manglar"),
    13: ("#800080", "Bosque plantado"),
    14: ("#40E0D0", "Humedal"),
    15: ("#F4A460", "Terreno arenoso"),
    16: ("#696969", "Terreno rocoso"),
    17: ("#FF6347", "Cultivos permanentes"),
    18: ("#FFA500", "Cultivos transitorios"),
    19: ("#9ACD32", "Pastos limpios"),
    20: ("#6B8E23", "Pastos con arboles"),
    21: ("#20B2AA", "Zonas pantanosas"),
    22: ("#00CED1", "Vegetacion acuatica"),
}


def visualizar_prediccion(ruta_prediccion, ruta_salida=None):
    with rasterio.open(ruta_prediccion) as src:
        prediccion = src.read(1)

    clases_unicas = np.unique(prediccion)
    clases_unicas = clases_unicas[clases_unicas > 0]

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    from matplotlib.colors import ListedColormap
    colores = [COLORES_IDEAM.get(c, ("#000000", f"Clase {c}"))[0] for c in range(max(clases_unicas) + 1)]
    cmap = ListedColormap(colores)

    im = ax.imshow(prediccion, cmap=cmap, vmin=0, vmax=max(clases_unicas))
    ax.set_title("Cobertura del Suelo - Armenia, Quindio", fontsize=14, fontweight="bold")
    ax.set_xlabel("Columna")
    ax.set_ylabel("Fila")

    patches = [
        mpatches.Patch(color=COLORES_IDEAM[c][0], label=COLORES_IDEAM[c][1])
        for c in clases_unicas if c in COLORES_IDEAM
    ]
    ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)

    plt.tight_layout()

    if ruta_salida:
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        print(f"Mapa guardado: {ruta_salida}")
    else:
        plt.show()

    plt.close()


def comparar_prediccion(ruta_imagen, ruta_etiquetas, ruta_prediccion, ruta_salida=None):
    with rasterio.open(ruta_imagen) as src:
        imagen = src.read([3, 2, 1])
        imagen = np.transpose(imagen, (1, 2, 0))
        imagen = np.clip(imagen / np.percentile(imagen, 98), 0, 1)

    with rasterio.open(ruta_etiquetas) as src:
        etiquetas = src.read(1)

    with rasterio.open(ruta_prediccion) as src:
        prediccion = src.read(1)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].imshow(imagen)
    axes[0].set_title("Sentinel-2 (RGB)")
    axes[0].axis("off")

    from matplotlib.colors import ListedColormap
    max_cls = max(etiquetas.max(), prediccion.max())
    colores = [COLORES_IDEAM.get(c, ("#000000", f"Clase {c}"))[0] for c in range(max_cls + 1)]
    cmap = ListedColormap(colores)

    axes[1].imshow(etiquetas, cmap=cmap, vmin=0, vmax=max_cls)
    axes[1].set_title("Etiquetas IDEAM")
    axes[1].axis("off")

    axes[2].imshow(prediccion, cmap=cmap, vmin=0, vmax=max_cls)
    axes[2].set_title("Prediccion del Modelo")
    axes[2].axis("off")

    plt.tight_layout()

    if ruta_salida:
        plt.savefig(ruta_salida, dpi=300, bbox_inches="tight")
        print(f"Comparacion guardada: {ruta_salida}")
    else:
        plt.show()

    plt.close()


def main():
    root = Path(__file__).parent.parent
    with open(root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ruta_prediccion = Path(config["paths"]["output"]) / "prediccion_armenia.tif"
    ruta_imagen = Path(config["paths"]["raw"]) / "sentinel2_armenia_2024.tif"
    ruta_etiquetas = Path(config["paths"]["labels"]) / "etiquetas_armenia.tif"
    ruta_salida = Path(config["paths"]["output"]) / "mapa_coberturas.png"
    ruta_comparacion = Path(config["paths"]["output"]) / "comparacion.png"

    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    if ruta_prediccion.exists():
        visualizar_prediccion(str(ruta_prediccion), str(ruta_salida))

        if ruta_imagen.exists() and ruta_etiquetas.exists():
            comparar_prediccion(str(ruta_imagen), str(ruta_etiquetas), str(ruta_prediccion), str(ruta_comparacion))
    else:
        print(f"No se encontro la prediccion: {ruta_prediccion}")
        print("Primero ejecute predict.py")


if __name__ == "__main__":
    main()
