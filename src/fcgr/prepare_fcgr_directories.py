# ============================================================
# PREPARE_PYTORCH_DATASET.PY
# ============================================================
#
# Convierte los conjuntos train/val/test generados
# previamente en una estructura compatible con
# torchvision.datasets.ImageFolder.
#
# Estructura final:
#
#   fcgr_512_by_classes/
#   ├── train/
#   │   ├── 0/
#   │   └── 1/
#   ├── val/
#   │   ├── 0/
#   │   └── 1/
#   └── test/
#       ├── 0/
#       └── 1/
#
# Cada imagen se copia automáticamente a la carpeta
# correspondiente según su etiqueta.
#
# ============================================================


# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import os
import shutil

import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Carpeta que contiene train.csv, val.csv y test.csv

SPLIT_DIR = "../data/fcgr_512_split"

# Columna con la ruta de la imagen

IMG_COL = "png_path"

# Columna con la etiqueta original

LABEL_COL = "label"

# Carpeta de salida compatible con PyTorch

OUT_DIR = "../data/fcgr_512_by_classes"

os.makedirs(
    OUT_DIR,
    exist_ok=True
)


# ============================================================
# CARGA DE LOS SPLITS
# ============================================================

train_df = pd.read_csv(
    os.path.join(
        SPLIT_DIR,
        "train.csv"
    )
)

val_df = pd.read_csv(
    os.path.join(
        SPLIT_DIR,
        "val.csv"
    )
)

test_df = pd.read_csv(
    os.path.join(
        SPLIT_DIR,
        "test.csv"
    )
)


# ============================================================
# NORMALIZACIÓN DE RUTAS
# ============================================================

def normalize_path(p):
    """
    Normaliza las rutas de las imágenes para que
    puedan encontrarse correctamente desde src/.
    """

    p = str(p)

    if os.path.exists(p):

        return p

    p2 = os.path.join("..", p)

    return p2


train_df["filepath"] = train_df[IMG_COL].map(
    normalize_path
)

val_df["filepath"] = val_df[IMG_COL].map(
    normalize_path
)

test_df["filepath"] = test_df[IMG_COL].map(
    normalize_path
)


# ============================================================
# MAPEO DE ETIQUETAS
# ============================================================

unique_labels = sorted(

    pd.concat([

        train_df[LABEL_COL].astype(str),

        val_df[LABEL_COL].astype(str),

        test_df[LABEL_COL].astype(str)

    ]).unique()

)

label2idx = {

    lab: i

    for i, lab in enumerate(unique_labels)

}

print("\nEtiquetas mapeadas a índices:")
print(label2idx)
print()


train_df["label_idx"] = (
    train_df[LABEL_COL]
    .astype(str)
    .map(label2idx)
)

val_df["label_idx"] = (
    val_df[LABEL_COL]
    .astype(str)
    .map(label2idx)
)

test_df["label_idx"] = (
    test_df[LABEL_COL]
    .astype(str)
    .map(label2idx)
)


# ============================================================
# CREACIÓN DE ESTRUCTURA IMAGEFOLDER
# ============================================================

for split, df in [

    ("train", train_df),

    ("val", val_df),

    ("test", test_df)

]:

    split_dir = os.path.join(
        OUT_DIR,
        split
    )

    os.makedirs(
        split_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Crear carpetas de clase
    # --------------------------------------------------------

    for i in range(len(unique_labels)):

        os.makedirs(

            os.path.join(
                split_dir,
                str(i)
            ),

            exist_ok=True

        )

    # --------------------------------------------------------
    # Copiar imágenes
    # --------------------------------------------------------

    for _, row in df.iterrows():

        src = row["filepath"]

        cls = row["label_idx"]

        dst = os.path.join(

            split_dir,

            str(cls),

            os.path.basename(src)

        )

        if not os.path.exists(src):

            print(
                f"WARNING: Imagen no encontrada: {src}"
            )

            continue

        if not os.path.exists(dst):

            shutil.copy(
                src,
                dst
            )

    print(
        f"Split '{split}' copiado en {split_dir}"
    )


# ============================================================
# RESUMEN FINAL
# ============================================================

print("\nESTRUCTURA COMPLETADA CORRECTAMENTE EN:")

print(OUT_DIR)
