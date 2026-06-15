# ============================================================
# PREPARE_NNRTIDF_EFV.PY
# ============================================================
#
# Carga el dataset NNRTI de HIVDB, filtra las muestras
# correspondientes a EFV (Efavirenz) y reconstruye las
# secuencias proteicas a partir de las columnas P1..Pn.
#
# Durante el proceso:
#
# - Se eliminan caracteres no estándar.
# - Se reconstruyen las secuencias completas.
# - Se generan versiones truncadas a 512 y 300 aa.
#
# Columnas generadas:
#
#   Sequence_full
#   Sequence_512
#   Sequence_300
#
# Las posiciones "-" (consenso) se sustituyen por el
# aminoácido correspondiente del consenso cuando se
# proporciona un FASTA de referencia.
#
# ============================================================


# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

from pathlib import Path

import argparse
import re

from typing import Optional

import pandas as pd


# ============================================================
# CONSTANTES
# ============================================================

AA20 = set(list("ACDEFGHIKLMNPQRSTVWY"))


# ============================================================
# LECTURA DEL CONSENSO
# ============================================================

def read_fasta_one_seq(path: Path) -> str:
    """
    Lee un FASTA de una única secuencia y devuelve
    la secuencia concatenada.
    """

    seq = []

    with open(path, "r") as f:

        for line in f:

            if line.startswith(">"):
                continue

            seq.append(line.strip())

    return "".join(seq).upper()


# ============================================================
# RECONSTRUCCIÓN DE SECUENCIAS
# ============================================================

def reconstruct_sequence_from_Pcols(
    row: pd.Series,
    consensus: Optional[str]
) -> str:
    """
    Reconstruye una secuencia proteica a partir
    de las columnas P1..Pn.
    """

    pcols = [
        c for c in row.index
        if re.fullmatch(r"P\d+", c)
    ]

    pcols.sort(
        key=lambda c: int(c[1:])
    )

    seq = []

    for idx, col in enumerate(pcols, start=1):

        val = str(row[col]).strip()

        if not val or val == "nan":
            continue

        # Mezclas tipo I/V → tomar la primera

        if "/" in val:
            val = val.split("/")[0]

        # Posición consenso

        if val == "-":

            if consensus and len(consensus) >= idx:

                seq.append(
                    consensus[idx - 1]
                )

            else:
                continue

        # Caracteres especiales eliminados

        elif val in {".", "#", "~", "*"}:

            continue

        else:

            aa = val[0].upper()

            if aa in AA20:

                seq.append(aa)

    return "".join(seq)


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main():

    # --------------------------------------------------------
    # ARGUMENTOS DE ENTRADA
    # --------------------------------------------------------

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        required=True,
        help="Ruta al NNRTI_DataSet.txt"
    )

    ap.add_argument(
        "--out",
        required=True,
        help="Ruta CSV de salida"
    )

    ap.add_argument(
        "--consensus-fasta",
        help="FASTA del consenso de RT (recomendado)"
    )

    ap.add_argument(
        "--drug-col",
        default="Drug"
    )

    ap.add_argument(
        "--drug-name",
        default="EFV"
    )

    ap.add_argument(
        "--id-col",
        default="SeqID"
    )

    args = ap.parse_args()

    # --------------------------------------------------------
    # CARGA DEL DATASET
    # --------------------------------------------------------

    df = pd.read_csv(
        args.input,
        sep="\t",
        dtype=str
    ).fillna("")

    # --------------------------------------------------------
    # FILTRADO POR FÁRMACO
    # --------------------------------------------------------

    if args.drug_col in df.columns:

        df = df[
            df[args.drug_col]
            .str.upper()
            .str.contains(args.drug_name.upper())
        ]

    if df.empty:

        raise SystemExit(
            f"No se encontraron filas con {args.drug_name}"
        )

    # --------------------------------------------------------
    # CARGA DEL CONSENSO
    # --------------------------------------------------------

    consensus = None

    if args.consensus_fasta:

        consensus = read_fasta_one_seq(
            Path(args.consensus_fasta)
        )

        print(
            f"Consenso cargado "
            f"(longitud = {len(consensus)})"
        )

    # --------------------------------------------------------
    # RECONSTRUCCIÓN DE SECUENCIAS
    # --------------------------------------------------------

    out_rows = []

    for _, row in df.iterrows():

        seq_full = reconstruct_sequence_from_Pcols(
            row,
            consensus
        )

        if not seq_full:
            continue

        out_rows.append({

            "SeqID": row.get(args.id_col, ""),

            "Sequence_full": seq_full,

            "Sequence_512": seq_full[:512],

            "Sequence_300": seq_full[:300]

        })

    # --------------------------------------------------------
    # EXPORTACIÓN DEL DATASET
    # --------------------------------------------------------

    out_df = pd.DataFrame(out_rows)

    out_df.to_csv(
        args.out,
        index=False
    )

    print(
        f"Secuencias EFV escritas en "
        f"{args.out} (n={len(out_df)})"
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
python src/data/prepare_nnrtidf_efv.py \
    --input data/NNRTI_DataSet.txt \
    --out data/efv_sequences.csv \
    --consensus-fasta data/consensus_RT_B.fasta
"""
