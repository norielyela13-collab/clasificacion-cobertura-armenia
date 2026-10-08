import yaml
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from pathlib import Path
from shapely.geometry import mapping, shape
import geopandas as gpd
import ee


MAPEADO_IDEAM = {
    1: 1,   # Bosque denso
    2: 2,   # Bosque fragmentado
    3: 3,   # Bosque de galeria
    4: 4,   # Bosque secundario
    5: 5,   # Pastos
    6: 6,   # Agricultura
    7: 7,   # Zonas urbanas
    8: 8,   # Cuerpos de agua
    9: 9,   # Zonas desnudas
    10: 10, # Nubes
    11: 11, # Paramo
    12: 12, # Manglar
    13: 13, # Bosque plantado
    14: 14, # Humedal
    15: 15, # Terreno arenoso
    16: 16, # Terreno rocoso
    17: 17, # Cultivos permanentes
    18: 18, # Cultivos transitorios
    19: 19, # Pastos limpios
    20: 20, # Pastos con arboles
    21: 21, # Zonas pantanosas
    22: 22, # Vegetacion acuatica
}


def descargar_ideam_gee(aoi, output_path):
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    ideam = ee.Image("projects/sat-io/open-datasets/IDEAM/COBERTURA_TIERRA_2018")
    ideam_clip = ideam.clip(aoi)

    url = ideam_clip.getDownloadURL({
        "region": aoi,
        "scale": 10,
        "format": "GEO_TIFF",
        "crs": "EPSG:4326"
    })

    import requests
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"IDEAM descargado: {output_path}")
    return output_path


def alinear_etiquetas(ruta_imagen, ruta_etiquetas, ruta_salida):
    with rasterio.open(ruta_imagen) as src_img:
        perfil_img = src_img.profile
        bounds_img = src_img.bounds
        transform_img = src_img.transform
        width = src_img.width
        height = src_img.height

    etiquetas_alineadas = np.zeros((height, width), dtype=np.uint8)

    with rasterio.open(ruta_etiquetas) as src_etiquetas:
        reproject(
            source=rasterio.band(src_etiquetas, 1),
            destination=etiquetas_alineadas,
            src_transform=src_etiquetas.transform,
            src_crs=src_etiquetas.crs,
            dst_transform=transform_img,
            dst_crs=perfil_img["crs"],
            resampling=Resampling.nearest
        )

    perfil = perfil_img.copy()
    perfil.update(
        dtype="uint8",
        count=1,
        compress="lzw",
        nodata=0
    )

    with rasterio.open(ruta_salida, "w", **perfil) as dst:
        dst.write(etiquetas_alineadas, 1)

    print(f"Etiquetas alineadas guardadas: {ruta_salida}")
    return ruta_salida


def generar_desde_shapefile(ruta_imagen, ruta_shapefile, ruta_salida, campo_clase="clase"):
    gdf = gpd.read_file(ruta_shapefile)
    gdf = gdf.to_crs(epsg=4326)

    with rasterio.open(ruta_imagen) as src:
        perfil = src.profile.copy()
        perfil.update(dtype="uint8", count=1, compress="lzw", nodata=0)
        shape = (src.height, src.width)
        transform = src.transform

    from rasterio import features
    shapes = ((geom, int(clase)) for geom, clase in zip(gdf.geometry, gdf[campo_clase]))
    etiquetas = features.rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8")

    with rasterio.open(ruta_salida, "w", **perfil) as dst:
        dst.write(etiquetas, 1)

    print(f"Etiquetas desde shapefile: {ruta_salida}")
    return ruta_salida


def main():
    root = Path(__file__).parent.parent
    with open(root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ruta_imagen = Path(config["paths"]["raw"]) / "sentinel2_armenia_2024.tif"
    ruta_etiquetas_ideam = Path(config["paths"]["labels"]) / "ideam_armenia.tif"
    ruta_salida = Path(config["paths"]["labels"]) / "etiquetas_armenia.tif"

    if not ruta_imagen.exists():
        print("Primero ejecute download_sentinel.py para descargar la imagen")
        return

    if not ruta_etiquetas_ideam.exists():
        print("Descargando mapa IDEAM desde Google Earth Engine...")
        ee.Initialize()
        centro = config["zona"]["centroide"]
        radio = config["zona"]["radio_km"] * 1000
        aoi = ee.Geometry.Point(centro).buffer(radio).bounds()
        descargar_ideam_gee(aoi, str(ruta_etiquetas_ideam))

    alinear_etiquetas(str(ruta_imagen), str(ruta_etiquetas_ideam), str(ruta_salida))


if __name__ == "__main__":
    main()
