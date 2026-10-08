# Bitácora de Desarrollo - Clasificación de Cobertura Terrestre

## Prompts del Usuario

### Prompt 1
> Crear un programa en py que me permita clasificar o detectar la cobertura terreste de una zona de colombia empleando modelos de ia de codigo abierto.

### Prompt 2
> Crear un programa en py que me permita clasificar o detectar el tipo de cobertura terreste de una zona de colombia empleando modelos de ia de codigo abierto.

### Prompt 3
> Crear un programa en py que me permita clasificar o detectar el tipo de cobertura terreste de una zona de colombia empleando modelos de ia de codigo abierto.

### Prompt 4 (Respuestas a preguntas)
- Fuente de imágenes: Sentinel-2
- Enfoque IA: Segmentación semántica
- Datos etiquetados: No, necesita generarlos
- Clases: Cobertura IDEAM (23 categorías)

### Prompt 5
> 1, usa google, 2, armenia, 3, si

Decisiones:
- Google Earth Engine para descarga
- Armenia, Quindío como zona de interés
- GPU disponible para entrenamiento

### Prompt 6
> si, procede

### Prompt 7
> sal tu xd

### Prompt 8
> conectate a chat gpt

### Prompt 9
> ok. ya cambie a build, desarrolla lo que te dije, pero no te conectes al chat gpt

### Prompt 10
> asi mismo crea un archivo bitacora que consigne todas las decisiones y promps utilizados

---

## Decisiones Técnicas

| # | Decisión | Justificación |
|---|----------|---------------|
| 1 | Sentinel-2 como fuente | Gratuito, 10m resolución, ideal para Colombia |
| 2 | Google Earth Engine | Acceso API gratuito, catálogo completo Sentinel-2 |
| 3 | Armenia, Quindío | Zona urbana/rural mixta, buena diversidad de coberturas |
| 4 | SegFormer B2 | Estado del arte en segmentación, eficiente con GPU |
| 5 | Etiquetas IDEAM | Mapa oficial de coberturas de Colombia, 23 clases |
| 6 | Parches 256×256 | Tamaño estándar para SegFormer, balance memoria/rendimiento |
| 7 | Solapamiento 50% | Aumenta datos de entrenamiento, reduce bordes |
| 8 | Loss: CE + Dice | CrossEntropy para clasificación + Dice para segmentación |
| 9 | Optimizador AdamW | Estándar para transformers, weight decay integrado |
| 10 | Early stopping (patience=10) | Evita overfitting |

---

## Estructura Creada

```
Taller/
├── config/config.yaml
├── data/{raw,labels,patches}/
├── src/
│   ├── download_sentinel.py
│   ├── generate_labels.py
│   ├── prepare_dataset.py
│   ├── train.py
│   ├── predict.py
│   └── visualize.py
├── label_tool/label_app.py
├── models/
├── requirements.txt
├── README.md
└── BITACORA.md
```

---

## Estado de Instalación

- [x] Estructura de directorios
- [x] requirements.txt
- [x] config/config.yaml
- [x] src/download_sentinel.py
- [x] src/generate_labels.py
- [x] src/prepare_dataset.py
- [x] src/train.py
- [x] src/predict.py
- [x] src/visualize.py
- [x] label_tool/label_app.py
- [x] README.md
- [x] BITACORA.md
- [x] Instalación de dependencias (completada)

---

## Fecha
07 de Octubre de 2026
