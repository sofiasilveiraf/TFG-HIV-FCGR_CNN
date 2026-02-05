"""
Crea un índice (CSV) para entrenar la CNN a partir de las imágenes FCGR.
Suposiciones:
- Tienes un CSV limpio con etiquetas: data/efv_sequences_labeled_clean.csv
 con al menos las columnas: SeqID, label
- Las imágenes FCGR están en un directorio (p.ej. data/fcgr_images)
 y se llaman así:
   SeqID_512_sf20_res200.png
   SeqID_300_sf20_res200.png
Genera:
- Un CSV con columnas: id, png_path, label
 que usaremos para el Dataset y para hacer los splits.
"""

import argparse

from pathlib import Path

import pandas as pd

import csv

def main():

    ap = argparse.ArgumentParser(description="Crear índice FCGR para longitud 300 o 512.")

    ap.add_argument("--input", required=True, help="CSV con SeqID y label")

    ap.add_argument("--img-dir", required=True, help="Directorio con imágenes FCGR")

    ap.add_argument("--length", type=int, choices=[300, 512], required=True,

                    help="Longitud de secuencia usada en la imagen")

    ap.add_argument("--out", required=True, help="CSV de salida")

    args = ap.parse_args()

    df = pd.read_csv(args.input)

    if "SeqID" not in df.columns or "label" not in df.columns:

        raise SystemExit("El CSV de entrada debe tener columnas 'SeqID' y 'label'.")

    img_dir = Path(args.img_dir)

    rows = []

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

    out_df = pd.DataFrame(rows)

    # Fix: forzar todo a string para evitar errores al guardar CSV

    out_df = out_df.astype(str)

    # Fix: guardar siempre con comillas (QUOTE_ALL)

    out_df.to_csv(args.out, index=False, quoting=csv.QUOTE_ALL)

    print(f"\n Índice FCGR guardado en: {args.out}")

    print(f"   Muestras incluidas: {len(out_df)}")


if __name__ == "__main__":

    main()
 

"""
Ejecutar así:
# 512 aa

python src/cgr_fcgr/make_index_fcgr_len.py \
 --input data/efv_sequences_labeled_clean.csv \
 --img-dir data/fcgr_images \
 --length 512 \
 --out data/fcgr/index_512.csv

# 300 aa
python src/cgr_fcgr/make_index_fcgr_len.py \
 --input data/efv_sequences_labeled_clean.csv \
 --img-dir data/fcgr_images \
 --length 300 \
 --out data/fcgr/index_300.csv
 """
