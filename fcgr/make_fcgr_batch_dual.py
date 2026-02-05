#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Genera imágenes FCGR para cada secuencia (dos variantes por fila):

- FCGR usando la secuencia completa (columna Sequence_512)

- FCGR usando la secuencia recortada a 300 aa (columna Sequence_300)

Usa n-flake óptimo (sf≈0.86327) para 20 aa y resolución configurable.

Ejemplo de uso:

    python src/fcgr/make_fcgr_batch_dual.py \

        --input data/efv_sequences_labeled_clean.csv \

        --outdir data/fcgr_images \

        --res 200

"""

from __future__ import annotations

import argparse

from math import pi, sin

from pathlib import Path

from typing import Dict, Tuple

import numpy as np

import pandas as pd

from PIL import Image

from tqdm import tqdm

AA = "ARNDCQEGHILKMFPSTWYV"  # alfabeto de 20 aa


def compute_sf20(n: int = 20) -> float:

    """Scaling factor óptimo para n divisible por 4 (n=20 para aminoácidos)."""

    if n % 4 != 0:

        raise ValueError("This sf20 formula assumes n divisible by 4 (use n=20 for amino acids).")

    m = n // 4

    r_ratio = sin(pi / n) / (sin(pi / n) + sin(pi / n + 2 * pi * m / n))

    return 1.0 - r_ratio


def polygon_vertices(n: int, radius: float = 1.0, angle_offset: float = -pi / 2) -> np.ndarray:

    """Vértices (x,y) de un n-ágono regular centrado."""

    angles = angle_offset + 2 * pi * np.arange(n) / n

    x = radius * np.cos(angles)

    y = radius * np.sin(angles)

    return np.stack([x, y], axis=1)


def fcgr_matrix(

    sequence: str,

    alphabet: str = AA,

    res: int = 200,

    sf: float = 0.86,

    angle_offset: float = -pi / 2,

) -> np.ndarray:

    """Genera la matriz FCGR normalizada en [0,1] para una secuencia."""

    seq = (sequence or "").strip().upper()

    alpha_index: Dict[str, int] = {aa: i for i, aa in enumerate(alphabet)}

    verts = polygon_vertices(len(alphabet), 1.0, angle_offset)

    x = 0.0

    y = 0.0

    grid = np.zeros((res, res), dtype=np.float64)

    def to_idx(v: float) -> int:

        idx = int((v + 1.0) * 0.5 * (res - 1))

        return 0 if idx < 0 else (res - 1 if idx >= res else idx)

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


def save_png(grid: np.ndarray, outfile: Path) -> None:

    """Guarda la matriz FCGR como PNG en escala de grises."""

    outfile.parent.mkdir(parents=True, exist_ok=True)

    arr8 = (np.clip(grid, 0, 1) * 255).astype(np.uint8)

    Image.fromarray(arr8, mode="L").save(str(outfile))


def process_csv(input_csv: Path, outdir: Path, res: int = 200) -> Tuple[int, int]:

    """Procesa el CSV y genera imágenes para Sequence_512 y Sequence_300."""

    df = pd.read_csv(input_csv)

    # Detecta columnas de secuencia disponibles

    seq_cols = [c for c in ("Sequence_512", "Sequence_300", "Sequence_full", "Sequence") if c in df.columns]

    if not seq_cols:

        raise ValueError("No encuentro columnas de secuencia (esperaba Sequence_512 / Sequence_300 / Sequence_full / Sequence).")

    # Solo trabajamos con las dos esperadas; si falta alguna se salta

    want_cols = [c for c in ("Sequence_512", "Sequence_300") if c in seq_cols]

    if not want_cols:

        # fallback: si solo hay 'Sequence' la usamos como 512

        want_cols = ["Sequence"]

    sf20 = compute_sf20(20)

    alpha = AA

    n512 = 0

    n300 = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="FCGR"):

        seq_id = str(row.get("SeqID", f"seq{idx+1:05d}"))

        # 512

        if "Sequence_512" in want_cols:

            s = str(row["Sequence_512"])

            grid = fcgr_matrix(s, alphabet=alpha, res=res, sf=sf20)

            save_png(grid, outdir / f"{seq_id}_512_sf20_res{res}.png")

            n512 += 1

        # 300

        if "Sequence_300" in want_cols:

            s = str(row["Sequence_300"])

            grid = fcgr_matrix(s, alphabet=alpha, res=res, sf=sf20)

            save_png(grid, outdir / f"{seq_id}_300_sf20_res{res}.png")

            n300 += 1

        # fallback si solo hay 'Sequence'

        if want_cols == ["Sequence"]:

            s = str(row["Sequence"])

            grid = fcgr_matrix(s, alphabet=alpha, res=res, sf=sf20)

            save_png(grid, outdir / f"{seq_id}_seq_sf20_res{res}.png")

            n512 += 1  # cuenta como “una variante”

    return n512, n300


def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Batch FCGR (sf≈0.863) para 512 y 300 aa.")

    p.add_argument("--input", required=True, type=Path, help="CSV con columnas Sequence_512 / Sequence_300 (y opcionalmente SeqID).")

    p.add_argument("--outdir", required=True, type=Path, help="Carpeta de salida para PNGs.")

    p.add_argument("--res", type=int, default=200, help="Resolución de la imagen (default=200).")

    return p.parse_args()


def main() -> None:

    args = parse_args()

    n512, n300 = process_csv(args.input, args.outdir, res=args.res)

    print(f"Listo. Generadas: 512={n512}, 300={n300}. Salida: {args.outdir}")


if __name__ == "__main__":

    main()
 
"""
Ejecutar así:

    source .venv/bin/activate
python src/fcgr/make_fcgr_batch_dual.py \
 --input data/efv_sequences_labeled_clean.csv \
 --outdir data/fcgr_images \
 --res 200
"""