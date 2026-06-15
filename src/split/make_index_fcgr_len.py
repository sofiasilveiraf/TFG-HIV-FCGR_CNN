# ============================================================
# MAKE_INDEX_FCGR_LEN.PY
# ============================================================
#
# Crea un índice (CSV) para entrenar la CNN a partir de las
# imágenes FCGR generadas previamente.
#
# Suposiciones:
#
# - Existe un CSV limpio con etiquetas:
#       data/efv_sequences_labeled_clean.csv
#
# - El CSV contiene al menos las columnas:
#       SeqID
#       label
#
# - Las imágenes FCGR se encuentran en un directorio y siguen
#   la nomenclatura:
#
#       SeqID_512_sf20_res200.png
#       SeqID_300_sf20_res200.png
#
# Salida:
#
# - CSV con las columnas:
#       id
#       png_path
#       label
#
# Este archivo se utilizará posteriormente para construir
# el Dataset y realizar los splits de entrenamiento,
# validación y prueba.
#
# ============================================================


# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import argparse
import csv

from pathlib import Path

import pandas as pd


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # ARGUMENTOS DE ENTRADA
    # --------------------------------------------------------

    ap = argparse.ArgumentParser(
        description="Crear índice FCGR para longitud 300 o 512."
    )

    ap.add_argument(
        "--input",
        required=True,
        help="CSV con SeqID y label"
    )

    ap.add_argument(
        "--img-dir",
        required=True,
        help="Directorio con imágenes FCGR"
    )

    ap.add_argument(
        "--length",
        type=int,
        choices=[300, 512],
        required=True,
        help="Longitud de secuencia usada en la imagen"
    )

    ap.add_argument(
        "--out",
        required=True,
        help="CSV de salida"
    )

    args = ap.parse_args()

    # --------------------------------------------------------
    # CARGA DEL DATASET
    # --------------------------------------------------------

    df = pd.read_csv(args.input)

    if "SeqID" not in df.columns or "label" not in df.columns:

        raise SystemExit(
            "El CSV de entrada debe tener columnas "
            "'SeqID' y 'label'."
        )

    img_dir = Path(args.img_dir)

    rows = []

    # --------------------------------------------------------
    # CONSTRUCCIÓN DEL ÍNDICE
    # --------------------------------------------------------

    for _, row in df.iterrows():

        sid = str(row["SeqID"])

        fname = f"{sid}_{args.length}_sf20_res200.png"

        fpath = img_dir / fname

        if not fpath.exists():

            print(f"Imagen no encontrada: {fpath}")

            continue

        rows.append({

            "id": sid,

            "png_path": str(fpath),

            "label": int(row["label"])

        })

    # --------------------------------------------------------
    # CREACIÓN DEL DATAFRAME FINAL
    # --------------------------------------------------------

    out_df = pd.DataFrame(rows)

    # Fix: forzar todo a string para evitar errores al guardar CSV

    out_df = out_df.astype(str)

    # Fix: guardar siempre con comillas (QUOTE_ALL)

    out_df.to_csv(
        args.out,
        index=False,
        quoting=csv.QUOTE_ALL
    )

    print(f"\nÍndice FCGR guardado en: {args.out}")
    print(f"Muestras incluidas: {len(out_df)}")


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":

    main()


# ============================================================
# EJEMPLOS DE EJECUCIÓN
# ============================================================

"""
------------------------------------------------------------
ÍNDICE PARA SECUENCIAS DE 512 AMINOÁCIDOS
------------------------------------------------------------

python src/cgr_fcgr/make_index_fcgr_len.py \
    --input data/efv_sequences_labeled_clean.csv \
    --img-dir data/fcgr_images \
    --length 512 \
    --out data/fcgr/index_512.csv


------------------------------------------------------------
ÍNDICE PARA SECUENCIAS DE 300 AMINOÁCIDOS
------------------------------------------------------------

python src/cgr_fcgr/make_index_fcgr_len.py \
    --input data/efv_sequences_labeled_clean.csv \
    --img-dir data/fcgr_images \
    --length 300 \
    --out data/fcgr/index_300.csv
"""
