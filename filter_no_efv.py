"""
filter_no_efv.py
----------------
Filtra un CSV (efv_sequences_labeled.csv) eliminando las filas sin valor
numérico de EFV.
Mantiene todas las columnas originales y guarda un nuevo CSV filtrado.
"""
import argparse
import pandas as pd

def main():
   parser = argparse.ArgumentParser(description="Elimina filas sin valor de EFV numérico.")
   parser.add_argument("--input", required=True, help="Archivo CSV de entrada (p.ej. efv_sequences_labeled.csv)")
   parser.add_argument("--output", required=True, help="Archivo CSV de salida filtrado")
   parser.add_argument("--col", default="EFV", help="Nombre de la columna FOLD (por defecto EFV)")
   args = parser.parse_args()
   # Leer CSV
   df = pd.read_csv(args.input)
   print(f" Archivo cargado: {args.input} ({len(df)} filas totales)")
   # Convertir EFV a numérico y filtrar
   df[args.col] = pd.to_numeric(df[args.col], errors="coerce")
   before = len(df)
   df = df[df[args.col].notna()].copy()
   after = len(df)
   # Guardar
   df.to_csv(args.output, index=False)
   print(f" Guardado: {args.output}")
   print(f" Filas eliminadas: {before - after} | Filas finales: {after}")

if __name__ == "__main__":
   main()


"""
Ejecutar así:
python3 src/data/filter_no_efv.py \
 --input data/efv_sequences_labeled.csv \
 --output data/efv_sequences_labeled_clean.csv \
 --col EFV
 """