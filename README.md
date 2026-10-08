# Clasificación de Cobertura Terrestre - Armenia, Quindío

Programa para clasificar la cobertura terrestre del municipio de Armenia (Quindio, Colombia) usando imagenes Sentinel-2 y el modelo de segmentacion semantica SegFormer.

## Requisitos

- Python 3.10+
- Cuenta de Google con Earth Engine habilitado
- GPU NVIDIA con CUDA (recomendado) o CPU

## Instalacion

```bash
pip install -r requirements.txt
```

## Configurar Google Earth Engine

```bash
earthengine authenticate
```

## Flujo de trabajo

### 1. Descargar imagen Sentinel-2

```bash
python -m src.download_sentinel
```

Descarga una imagen compuesta (mediana anual 2024) de Armenia con bandas B2, B3, B4, B8, B11, B12 e indices NDVI, NDWI, NDBI.

### 2. Generar etiquetas desde IDEAM

```bash
python -m src.generate_labels
```

Descarga el mapa de coberturas IDEAM y lo alinea con la imagen Sentinel-2.

### 3. Preparar dataset

```bash
python -m src.prepare_dataset
```

Crea parches de 256x256 con solapamiento del 50% y los divide en train/val/test.

### 4. Entrenar modelo

```bash
python -m src.train
```

Entrena SegFormer B2 con fine-tuning usando GPU. Guarda el mejor modelo en `models/best_model.pth`.

### 5. Predecir

```bash
python -m src.predict
```

Genera un mapa de coberturas para toda la zona de Armenia.

### 6. Visualizar resultados

```bash
python -m src.visualize
```

Genera mapas visuales y comparacion entre etiquetas IDEAM y prediccion del modelo.

## Estructura del proyecto

```
Taller/
├── config/config.yaml          # Configuracion general
├── data/
│   ├── raw/                    # Imagenes Sentinel-2
│   ├── labels/                 # Etiquetas IDEAM
│   └── patches/                # Parches de entrenamiento
├── src/                        # Codigo fuente
├── label_tool/                 # Herramienta de etiquetado
├── models/                     # Checkpoints del modelo
└── output/                     # Resultados
```

## Clases IDEAM (23 categorias)

| ID | Clase |
|----|-------|
| 1  | Bosque denso |
| 2  | Bosque fragmentado |
| 3  | Bosque de galeria |
| 4  | Bosque secundario |
| 5  | Pastos |
| 6  | Agricultura |
| 7  | Zonas urbanas |
| 8  | Cuerpos de agua |
| 9  | Zonas desnudas |
| 10 | Nubes |
| 11 | Paramo |
| 12 | Manglar |
| 13 | Bosque plantado |
| 14 | Humedal |
| 15 | Terreno arenoso |
| 16 | Terreno rocoso |
| 17 | Cultivos permanentes |
| 18 | Cultivos transitorios |
| 19 | Pastos limpios |
| 20 | Pastos con arboles |
| 21 | Zonas pantanosas |
| 22 | Vegetacion acuatica |

## Notas

- El mapa IDEAM se descarga desde Google Earth Engine (proyecto `sat-io`)
- El modelo SegFormer B2 esta preentrenado en ADE20K y se ajusta con las etiquetas IDEAM
- Para mejorar resultados, se recomienda corregir manualmente las etiquetas con la herramienta Streamlit
