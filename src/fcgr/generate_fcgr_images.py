# ============================================================
# MAKE_FCGR_BATCH_DUAL.PY
# ============================================================
#
# Genera imágenes FCGR para cada secuencia del dataset.
#
# Para cada muestra se generan dos variantes:
#
#   - FCGR usando Sequence_512
#   - FCGR usando Sequence_300
#
# Utiliza:
#
#   - Alfabeto de 20 aminoácidos
#   - Factor de escalado óptimo n-flake
#       sf ≈ 0.86327
#   - Resolución configurable
#
# Salida:
#
#   SeqID_512_sf20_res200.png
#   SeqID_300_sf20_res200.png
#
# ============================================================


# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

from __future__ import annotations

import argparse

from math import pi, sin

from pathlib import Path

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from PIL import Image

from tqdm import tqdm


# ============================================================
# CONSTANTES
# ============================================================

AA = "ARNDCQEGHILKMFPSTWYV"   # Alfabeto de 20 aminoácidos


# ============================================================
# FACTOR DE ESCALADO ÓPTIMO
# ============================================================

def compute_sf20(n: int = 20) -> float:
    """
    Calcula el scaling factor óptimo para un alfabeto
    de tamaño n divisible por 4 (n = 20 aminoácidos).
    """

    if n % 4 != 0:

        raise ValueError(
            "This sf20 formula assumes n divisible by 4 "
            "(use n=20 for amino acids)."
        )

    m = n // 4

    r_ratio = (
        sin(pi / n)
        /
        (
            sin(pi / n)
            + sin(pi / n + 2 * pi * m / n)
        )
    )

    return 1.0 - r_ratio


# ============================================================
# GENERACIÓN DE VÉRTICES DEL POLÍGONO
# ============================================================

def polygon_vertices(
    n: int,
    radius: float = 1.0,
    angle_offset: float = -pi / 2
) -> np.ndarray:
    """
    Genera los vértices (x,y) de un polígono regular
    centrado en el origen.
    """

    angles = (
        angle_offset
        + 2 * pi * np.arange(n) / n
    )

    x = radius * np.cos(angles)

    y = radius * np.sin(angles)

    return np.stack([x, y], axis=1)


# ============================================================
# GENERACIÓN DE MATRIZ FCGR
# ============================================================

def fcgr_matrix(

    sequence: str,

    alphabet: str = AA,

    res: int = 200,

    sf: float = 0.86,

    angle_offset: float = -pi / 2

) -> np.ndarray:

    """
    Genera una matriz FCGR normalizada
    en el intervalo [0,1].
    """

    seq = (sequence or "").strip().upper()

    alpha_index: Dict[str, int] = {
        aa: i
        for i, aa in enumerate(alphabet)
    }

    verts = polygon_vertices(
        len(alphabet),
        1.0,
        angle_offset
    )

    x = 0.0
    y = 0.0

    grid = np.zeros(
        (res, res),
        dtype=np.float64
    )

    def to_idx(v: float) -> int:

        idx = int(
            (v + 1.0)
            * 0.5
            * (res - 1)
        )

        return (
            0 if idx < 0
            else (res - 1 if idx >= res else idx)
        )

    for aa in seq:

        if aa not in alpha_index:
            continue

        vx, vy = verts[alpha_index[aa]]

        x = (1.0 - sf) * x + sf * vx
        y = (1.0 - sf) * y + sf * vy

        ix = to_idx(x)
        iy = to_idx(y)

        grid[res - 1 - iy, ix] += 1.0

    if grid.max() > 0:

        grid /= grid.max()

    return grid


# ============================================================
# GUARDAR IMAGEN PNG
# ============================================================

def save_png(
    grid: np.ndarray,
    outfile: Path
) -> None:

    """
    Guarda una matriz FCGR como imagen PNG
    en escala de grises.
    """

    outfile.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    arr8 = (
        np.clip(grid, 0, 1) * 255
    ).astype(np.uint8)

    Image.fromarray(
        arr8,
        mode="L"
    ).save(str(outfile))


# ============================================================
# PROCESAMIENTO DEL CSV
# ============================================================

def process_csv(
    input_csv: Path,
    outdir: Path,
    res: int = 200
) -> Tuple[int, int]:

    """
    Procesa un CSV y genera imágenes FCGR
    para las columnas Sequence_512 y Sequence_300.
    """

    df = pd.read_csv(input_csv)

    # --------------------------------------------------------
    # Detección de columnas disponibles
    # --------------------------------------------------------

    seq_cols = [

        c for c in (

            "Sequence_512",
            "Sequence_300",
            "Sequence_full",
            "Sequence"

        )

        if c in df.columns
    ]

    if not seq_cols:

        raise ValueError(
            "No encuentro columnas de secuencia "
            "(esperaba Sequence_512 / Sequence_300 / "
            "Sequence_full / Sequence)."
        )

    want_cols = [

        c for c in (
            "Sequence_512",
            "Sequence_300"
        )

        if c in seq_cols
    ]

    if not want_cols:

        want_cols = ["Sequence"]

    sf20 = compute_sf20(20)

    alpha = AA

    n512 = 0
    n300 = 0

    # --------------------------------------------------------
    # Generación de imágenes
    # --------------------------------------------------------

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="FCGR"
    ):

        seq_id = str(
            row.get(
                "SeqID",
                f"seq{idx+1:05d}"
            )
        )

        # ----------------------------------------------------
        # Sequence_512
        # ----------------------------------------------------

        if "Sequence_512" in want_cols:

            s = str(row["Sequence_512"])

            grid = fcgr_matrix(
                s,
                alphabet=alpha,
                res=res,
                sf=sf20
            )

            save_png(
                grid,
                outdir / f"{seq_id}_512_sf20_res{res}.png"
            )

            n512 += 1

        # ----------------------------------------------------
        # Sequence_300
        # ----------------------------------------------------

        if "Sequence_300" in want_cols:

            s = str(row["Sequence_300"])

            grid = fcgr_matrix(
                s,
                alphabet=alpha,
                res=res,
                sf=sf20
            )

            save_png(
                grid,
                outdir / f"{seq_id}_300_sf20_res{res}.png"
            )

            n300 += 1

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        if want_cols == ["Sequence"]:

            s = str(row["Sequence"])

            grid = fcgr_matrix(
                s,
                alphabet=alpha,
                res=res,
                sf=sf20
            )

            save_png(
                grid,
                outdir / f"{seq_id}_seq_sf20_res{res}.png"
            )

            n512 += 1

    return n512, n300


# ============================================================
# PARSEO DE ARGUMENTOS
# ============================================================

def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(
        description="Batch FCGR (sf≈0.863) para 512 y 300 aa."
    )

    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV con columnas Sequence_512 / Sequence_300"
    )

    p.add_argument(
        "--outdir",
        required=True,
        type=Path,
        help="Carpeta de salida para PNGs"
    )

    p.add_argument(
        "--res",
        type=int,
        default=200,
        help="Resolución de la imagen"
    )

    return p.parse_args()


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:

    args = parse_args()

    n512, n300 = process_csv(
        args.input,
        args.outdir,
        res=args.res
    )

    print(
        f"Listo. Generadas: "
        f"512={n512}, "
        f"300={n300}. "
        f"Salida: {args.outdir}"
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
source .venv/bin/activate

python src/fcgr/make_fcgr_batch_dual.py \
    --input data/efv_sequences_labeled_clean.csv \
    --outdir data/fcgr_images \
    --res 200
"""
