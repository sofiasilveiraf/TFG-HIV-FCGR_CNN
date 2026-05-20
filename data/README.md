# Data

Esta carpeta contiene las representaciones Frequency Chaos Game Representation (FCGR) generadas a partir de secuencias proteicas del VIH-1 utilizadas en este trabajo.

## Contenido

- `fcgr_512/`  
  Imágenes FCGR generadas con resolución 512 × 512 píxeles empleadas como entrada para el entrenamiento y evaluación de la red neuronal convolucional (CNN).

## Descripción

Las representaciones FCGR permiten transformar secuencias biológicas en estructuras espaciales que conservan patrones relevantes de información biológica y facilitan su procesamiento mediante técnicas de aprendizaje profundo.

Las imágenes almacenadas en este directorio fueron generadas a partir de secuencias de la transcriptasa inversa (RT) del VIH-1 obtenidas de la Stanford HIV Drug Resistance Database (HIVDB).

## Formato de nombres

Los archivos siguen el formato:

```text
<ID_secuencia>_512_sf20_res200.png
