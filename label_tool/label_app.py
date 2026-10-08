import streamlit as st
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import shape, mapping
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


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


def main():
    st.set_page_config(page_title="Etiquetado Coberturas", layout="wide")
    st.title("Herramienta de Etiquetado - Cobertura del Suelo")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Imagen Sentinel-2")
        ruta_imagen = st.text_input("Ruta imagen", "data/raw/sentinel2_armenia_2024.tif")

        if Path(ruta_imagen).exists():
            with rasterio.open(ruta_imagen) as src:
                imagen = src.read([3, 2, 1])
                imagen = np.transpose(imagen, (1, 2, 0))
                imagen = np.clip(imagen / np.percentile(imagen, 98), 0, 1)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(imagen)
            ax.set_title("Sentinel-2 RGB")
            ax.axis("off")
            st.pyplot(fig)
            plt.close()
        else:
            st.warning("Imagen no encontrada")

    with col2:
        st.subheader("Clase seleccionada")
        clases = {v[1]: k for k, v in COLORES_IDEAM.items() if k > 0}
        clase_nombre = st.selectbox("Clase", list(clases.keys()))
        clase_id = clases[clase_nombre]

        st.color_picker("Color", COLORES_IDEAM[clase_id][0], disabled=True)

        st.markdown("---")
        st.subheader("Leyenda")
        for k, (color, nombre) in COLORES_IDEAM.items():
            if k > 0:
                st.markdown(f":{color}[■] {nombre}")

    st.markdown("---")
    st.subheader("Exportar etiquetas")

    if st.button("Guardar etiquetas como GeoTIFF"):
        if Path(ruta_imagen).exists():
            with rasterio.open(ruta_imagen) as src:
                etiquetas = np.zeros((src.height, src.width), dtype=np.uint8)
                perfil = src.profile.copy()
                perfil.update(dtype="uint8", count=1, compress="lzw", nodata=0)

            ruta_salida = "data/labels/etiquetas_manuales.tif"
            Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)

            with rasterio.open(ruta_salida, "w", **perfil) as dst:
                dst.write(etiquetas, 1)

            st.success(f"Etiquetas guardadas: {ruta_salida}")
        else:
            st.error("Primero cargue una imagen valida")


if __name__ == "__main__":
    main()
