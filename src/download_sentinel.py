import os
import yaml
import ee
import geemap
import rasterio
import numpy as np
from rasterio.transform import from_bounds
from pathlib import Path


def inicializar_ee():
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()


def crear_zona_aoi(config):
    centro = config["zona"]["centroide"]
    radio = config["zona"]["radio_km"] * 1000
    punto = ee.Geometry.Point(centro)
    aoi = punto.buffer(radio).bounds()
    return aoi


def calcular_indices(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = image.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return image.addBands([ndvi, ndwi, ndbi])


def descargar_sentinel(config, output_path):
    inicializar_ee()
    aoi = crear_zona_aoi(config)

    coleccion = config["imagen"]["coleccion"]
    bandas = config["imagen"]["bandas"]
    fecha_inicio = config["imagen"]["fecha_inicio"]
    fecha_fin = config["imagen"]["fecha_fin"]
    nube_max = config["imagen"]["nube_max"]

    s2 = (
        ee.ImageCollection(coleccion)
        .filterBounds(aoi)
        .filterDate(fecha_inicio, fecha_fin)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", nube_max))
    )

    conteo = s2.size().getInfo()
    print(f"Imágenes encontradas: {conteo}")

    if conteo == 0:
        raise ValueError("No se encontraron imágenes Sentinel-2 para la zona y fechas especificadas")

    composicion = s2.median().clip(aoi)
    composicion = calcular_indices(composicion)

    todas_bandas = bandas + config["indices"]
    composicion = composicion.select(todas_bandas)

    print("Descargando imagen compuesta...")
    url = composicion.getDownloadURL({
        "region": aoi,
        "scale": config["imagen"]["resolucion"],
        "format": "GEO_TIFF",
        "crs": "EPSG:4326"
    })

    import requests
    response = requests.get(url, stream=True)
    response.raise_for_status()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Imagen guardada en: {output_path}")
    return output_path


def main():
    root = Path(__file__).parent.parent
    with open(root / "config" / "config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output = Path(config["paths"]["raw"]) / "sentinel2_armenia_2024.tif"
    descargar_sentinel(config, str(output))


if __name__ == "__main__":
    main()
