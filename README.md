# TFG-HIV-FCGR-CNN

Repositorio asociado al Trabajo de Fin de Grado:

**“Clasificación avanzada de secuencias proteicas mediante la representación del juego del caos y redes neuronales profundas”**

## Descripción

Este proyecto desarrolla un pipeline completo para la clasificación de resistencia a fármacos antirretrovirales del VIH-1 utilizando:

- Frequency Chaos Game Representation (FCGR)
- Redes neuronales convolucionales (CNN)
- Secuencias proteicas de la transcriptasa inversa (RT)

El conjunto de datos se obtuvo de la Stanford HIV Drug Resistance Database (HIVDB), seleccionando secuencias del subtipo B asociadas al fármaco efavirenz (EFV).

---

## Flujo de trabajo

1. Filtrado y limpieza de datos HIVDB
2. Reconstrucción de secuencias RT completas
3. Etiquetado binario según resistencia a EFV
4. Generación de imágenes FCGR
5. División train / validation / test
6. Entrenamiento de CNN en PyTorch
7. Evaluación del modelo

---

## Estructura del repositorio

- `data/` → datasets y representaciones FCGR
- `src/` → scripts de procesamiento y entrenamiento
- `models/` → modelos entrenados
- `results/` → métricas y figuras

---

## Tecnologías utilizadas

- Python
- PyTorch
- torchvision
- pandas
- NumPy
- scikit-learn
- matplotlib

---

## Dataset

Fuente:
Stanford HIV Drug Resistance Database (HIVDB)

https://hivdb.stanford.edu/

---

## Autora

Sofía Silveira Franco
Escola de Enxeñaría Industrial (UVigo)
