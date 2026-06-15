# ============================================================
# FILTER_NO_EFV.PY
# ============================================================
#
# Filtra un CSV eliminando las filas que no contienen
# un valor numérico válido para EFV.
#
# Entrada:
#
#   efv_sequences_labeled.csv
#
# Salida:
#
#   efv_sequences_labeled_clean.csv
#
# Se mantienen todas las columnas originales del dataset.
#
# ============================================================


# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import argparse

import pandas as pd


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # ARGUMENTOS DE ENTRADA
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="Elimina filas sin valor de EFV numérico."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Archivo CSV de entrada (p.ej. efv_sequences_labeled.csv)"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Archivo CSV de salida filtrado"
    )

    parser.add_argument(
        "--col",
        default="EFV",
        help="Nombre de la columna FOLD (por defecto EFV)"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # CARGA DEL DATASET
    # --------------------------------------------------------

    df = pd.read_csv(args.input)

    print(
        f"Archivo cargado: {args.input} "
        f"({len(df)} filas totales)"
    )

    # --------------------------------------------------------
    # FILTRADO DE VALORES EFV
    # --------------------------------------------------------

    df[args.col] = pd.to_numeric(
        df[args.col],
        errors="coerce"
    )

    before = len(df)

    df = df[
        df[args.col].notna()
    ].copy()

    after = len(df)

    # --------------------------------------------------------
    # EXPORTACIÓN DEL DATASET FILTRADO
    # --------------------------------------------------------

    df.to_csv(
        args.output,
        index=False
    )

    print(f"Guardado: {args.output}")

    print(
        f"Filas eliminadas: {before - after} | "
        f"Filas finales: {after}"
    )


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":

    main()


# ============================================================
# EJEMPLO DE EJECUCIÓN
# ============================================================

"""
python3 src/data/filter_no_efv.py \
    --input data/efv_sequences_labeled.csv \
    --output data/efv_sequences_labeled_clean.csv \
    --col EFV
"""
