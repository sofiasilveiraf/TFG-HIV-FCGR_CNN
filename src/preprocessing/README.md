# Preprocessing

Scripts utilizados para la preparación y limpieza de los datos obtenidos de la Stanford HIV Drug Resistance Database (HIVDB).

## Contenido

- `prepare_nnrtidf_efv.py`  
  Preparación y reconstrucción de secuencias RT asociadas al fármaco efavirenz (EFV).

- `filter_no_efv.py`  
  Filtrado de secuencias sin información válida de resistencia a EFV.

- `add_labels_efv.py`  
  Generación de etiquetas binarias de resistencia (0 = sensible, 1 = resistente).

## Objetivo

Estos scripts forman parte de la etapa de preprocesamiento del pipeline y permiten generar los datasets utilizados posteriormente para la creación de representaciones FCGR y el entrenamiento de la CNN.
