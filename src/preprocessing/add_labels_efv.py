# ============================================================
# ADD_LABELS_EFV.PY
# ============================================================
#
# Une un CSV de secuencias con el dataset NNRTI de Stanford
# y genera la etiqueta binaria:
#
#   0 = sensible
#   1 = resistente
#
# Reglas de etiquetado:
#
# 1) Si existe una columna SIR (por ejemplo EFV_SIR):
#
#       S → 0
#       R → 1
#       I → configurable mediante --treat_I_as
#
# 2) Si no existe una columna SIR utilizable:
#
#       EFV >= cutoff → resistente (1)
#       EFV < cutoff  → sensible (0)
#
# Salida:
#
#   CSV con:
#       SeqID
#       Sequence_*
#       label
#       EFV (opcional)
#
# ============================================================


# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# UTILIDADES
# ============================================================

def read_raw_tolerant(path: Path) -> pd.DataFrame:
    """
    Intenta leer NNRTI_DataSet utilizando distintos
    separadores y formatos.
    """

    path = Path(path)

    # --------------------------------------------------------
    # Intento TSV
    # --------------------------------------------------------

    try:
        return pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            low_memory=False
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Intento CSV estándar
    # --------------------------------------------------------

    try:
        return pd.read_csv(
            path,
            dtype=str,
            low_memory=False
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Detección automática
    # --------------------------------------------------------

    return pd.read_csv(
        path,
        sep=None,
        engine="python",
        dtype=str,
        low_memory=False
    )


def pick_sequence_column(
    df: pd.DataFrame,
    preferred: str | None
) -> str:
    """
    Selecciona la columna de secuencia que se utilizará.
    """

    if preferred and preferred in df.columns:
        return preferred

    candidates = [
        "Sequence",
        "Sequence_512",
        "Sequence_full",
        "Sequence_300",
        "Sequence_full_aa"
    ]

    for c in candidates:

        if c in df.columns:

            print(
                f"Usaré como columna de secuencia: {c}"
            )

            return c

    raise SystemExit(
        "No encuentro ninguna columna de secuencia. "
        "Esperaba una de: "
        "'Sequence', 'Sequence_512', "
        "'Sequence_300' o 'Sequence_full'."
    )


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # ARGUMENTOS DE ENTRADA
    # --------------------------------------------------------

    ap = argparse.ArgumentParser(
        description="Crea 'label' (0/1) combinando SIR/FOLD para EFV."
    )

    ap.add_argument(
        "--input",
        required=True,
        help="CSV de secuencias (con SeqID y Sequence_*)"
    )

    ap.add_argument(
        "--raw",
        required=True,
        help="Fichero NNRTI_DataSet.txt (Stanford)"
    )

    ap.add_argument(
        "--out",
        required=True,
        help="Ruta de salida CSV"
    )

    ap.add_argument(
        "--id-col",
        default="SeqID",
        help="Nombre de columna ID (por defecto: SeqID)"
    )

    ap.add_argument(
        "--seq-col",
        default=None,
        help="Columna de secuencia a usar (si no, se detecta)"
    )

    ap.add_argument(
        "--fold-col",
        default="EFV",
        help="Columna FOLD (por defecto: EFV)"
    )

    ap.add_argument(
        "--sir-col",
        default="EFV_SIR",
        help="Columna SIR (por defecto: EFV_SIR)"
    )

    ap.add_argument(
        "--cutoff",
        type=float,
        default=3.61,
        help="Umbral FOLD (>= resistente)"
    )

    ap.add_argument(
        "--treat_I_as",
        default="",
        choices=["", "0", "1", "drop"],
        help="Tratamiento de la categoría I"
    )

    ap.add_argument(
        "--drop-missing",
        action="store_true",
        help="Eliminar filas sin etiqueta"
    )

    args = ap.parse_args()

    idc = args.id_col
    fold = args.fold_col
    sir = args.sir_col

    # --------------------------------------------------------
    # CARGA DE SECUENCIAS
    # --------------------------------------------------------

    df_in = pd.read_csv(
        args.input,
        dtype=str
    )

    seq_col = pick_sequence_column(
        df_in,
        args.seq_col
    )

    df_in[idc] = df_in[idc].astype(str)

    # --------------------------------------------------------
    # CARGA DEL DATASET RAW
    # --------------------------------------------------------

    try:

        df_raw = read_raw_tolerant(args.raw)

    except Exception as e:

        raise SystemExit(
            f"No pude leer --raw correctamente.\n"
            f"Archivo: {args.raw}\n"
            f"Error: {e}"
        )

    # --------------------------------------------------------
    # COMPROBACIÓN DE COLUMNAS
    # --------------------------------------------------------

    missing = [
        c for c in (idc, fold, sir)
        if c not in df_raw.columns
    ]

    if missing:

        print(
            f"Aviso: faltan columnas en el raw: "
            f"{missing}"
        )

    cols_keep = [
        c for c in [idc, fold, sir]
        if c in df_raw.columns
    ]

    df_lab = df_raw[cols_keep].copy()

    df_lab[idc] = df_lab[idc].astype(str)

    if fold in df_lab.columns:

        df_lab[fold] = pd.to_numeric(
            df_lab[fold],
            errors="coerce"
        )

    if sir in df_lab.columns:

        df_lab[sir] = (
            df_lab[sir]
            .astype(str)
            .str.upper()
            .replace({"": pd.NA})
        )

    # --------------------------------------------------------
    # ETIQUETADO MEDIANTE SIR
    # --------------------------------------------------------

    label = None

    if (
        sir in df_lab.columns
        and df_lab[sir].notna().any()
    ):

        map_sir = {
            "S": 0,
            "R": 1,
            "I": pd.NA
        }

        treat = (
            args.treat_I_as or ""
        ).strip().lower()

        if treat == "0":
            map_sir["I"] = 0

        elif treat == "1":
            map_sir["I"] = 1

        elif treat == "drop":
            pass

        sir_up = (
            df_lab[sir]
            .astype(str)
            .str.upper()
        )

        label = sir_up.map(map_sir)

        if treat == "drop":

            mask_keep = ~sir_up.eq("I")

            df_lab = df_lab[mask_keep].copy()

            label = label[mask_keep]

    # --------------------------------------------------------
    # ETIQUETADO MEDIANTE FOLD
    # --------------------------------------------------------

    if (
        label is None
        or getattr(
            label,
            "isna",
            lambda: pd.Series([False])
        )().all()
    ):

        if fold not in df_lab.columns:

            raise SystemExit(
                "No hay SIR utilizable ni FOLD disponible."
            )

        label = (
            df_lab[fold] >= float(args.cutoff)
        ).astype("Int64")

    df_lab["label"] = label

    # --------------------------------------------------------
    # COLUMNAS PARA EL MERGE
    # --------------------------------------------------------

    keep_for_merge = [
        idc,
        "label"
    ]

    if fold in df_lab.columns:

        keep_for_merge.append(fold)

    df_lab_small = df_lab[
        keep_for_merge
    ].copy()

    # --------------------------------------------------------
    # UNIÓN CON LAS SECUENCIAS
    # --------------------------------------------------------

    merged = df_in.merge(
        df_lab_small,
        on=idc,
        how="left"
    )

    if args.drop_missing:

        merged = merged[
            merged["label"].notna()
        ].copy()

    # --------------------------------------------------------
    # GUARDAR RESULTADO
    # --------------------------------------------------------

    out_path = Path(args.out)

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    merged.to_csv(
        out_path,
        index=False
    )

    # --------------------------------------------------------
    # RESUMEN
    # --------------------------------------------------------

    n = len(merged)

    has_labels = merged["label"].notna()

    n_lab = int(has_labels.sum())

    pct_resist = (
        float(
            merged.loc[
                has_labels,
                "label"
            ].mean() * 100.0
        )
        if n_lab
        else 0.0
    )

    print(f"Guardado con etiquetas: {out_path}")

    print(
        f"n={n} | "
        f"con_label={n_lab} | "
        f"%resist={pct_resist:.2f}"
    )

    if fold in merged.columns:

        print(
            f"(Incluida columna FOLD: {fold})"
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
python3 src/data/add_labels_efv.py \
    --input data/efv_sequences.csv \
    --raw data/NNRTI_DataSet.txt \
    --out data/efv_sequences_labeled.csv \
    --id-col SeqID \
    --seq-col Sequence_512 \
    --fold-col EFV \
    --cutoff 3.61 \
    --drop-missing
"""
