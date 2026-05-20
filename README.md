# TFG-HIV-FCGR-CNN

Clasificación avanzada de secuencias proteicas mediante representaciones FCGR y redes neuronales convolucionales.

## Descripción

Este repositorio contiene el código asociado al Trabajo de Fin de Grado:

> “Clasificación avanzada de secuencias proteicas mediante la representación del juego del caos y redes neuronales profundas”

El proyecto desarrolla un pipeline completo para la clasificación de resistencia a fármacos antirretrovirales del VIH-1 utilizando:

- Frequency Chaos Game Representation (FCGR)
- Redes neuronales convolucionales (CNN)
- Secuencias proteicas de la transcriptasa inversa (RT)

El conjunto de datos utilizado se obtuvo de la Stanford HIV Drug Resistance Database (HIVDB), seleccionando secuencias del subtipo B asociadas al fármaco efavirenz (EFV).

---

## Flujo de trabajo

- Filtrado y limpieza de datos HIVDB
- Reconstrucción de secuencias RT completas
- Etiquetado binario según resistencia a EFV
- Generación de imágenes FCGR
- División del conjunto de datos en train / validation / test
- Entrenamiento de la CNN en PyTorch
- Evaluación del modelo

---

## Estructura del repositorio

- `data/` → datasets procesados y representaciones FCGR
- `src/` → scripts de procesamiento, entrenamiento y evaluación
- `models/` → modelos entrenados y checkpoints
- `results/` → métricas, gráficas y figuras obtenidas

---

## Tecnologías utilizadas

- Python
- PyTorch
- torchvision
- pandas
- NumPy
- scikit-learn
- matplotlib
- Biopython

---

## Dataset

Fuente:

Stanford HIV Drug Resistance Database (HIVDB)

https://hivdb.stanford.edu/

---

## Ejecución

Instalación de dependencias:

```bash
pip install -r requirements.txt
```

Entrenamiento del modelo:

```bash
python train_model.py
```

Evaluación del modelo:

```bash
python evaluate_model.py
```
---

## Referencia metodológica

La implementación de las representaciones FCGR se basa en el trabajo:

> Löchel, H. F., Eger, D., Sperlea, T., & Heider, D. (2019).  
> *Deep learning on chaos game representation for proteins*.  
> Bioinformatics, 36(1), 272–279.

https://doi.org/10.1093/bioinformatics/btz493
---

## Autora

Sofía Silveira Franco  
Grado en Ingeniería Biomédica  
Escola de Enxeñaría Industrial — Universidade de Vigo (UVigo)
