# ============================================================
# COMPARE_FCGR_IMAGES.PY
# ============================================================
#
# Comparativa visual de imágenes FCGR
#
# Muestra pares de imágenes FCGR generadas a partir de:
#
#   - Secuencias completas (512 aa)
#   - Secuencias truncadas (300 aa)
#
# Permite evaluar visualmente las diferencias estructurales
# entre ambas representaciones antes de seleccionar la
# longitud utilizada en el entrenamiento de la CNN.
#
# ============================================================


# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import os
import random

from pathlib import Path

import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# CONFIGURACIÓN
# ============================================================

# Carpeta donde se almacenan las imágenes FCGR

IMG_DIR = Path("data/fcgr_images")

# Número de pares a visualizar

N = 6

# Extensión de las imágenes

EXT = ".png"


# ============================================================
# BÚSQUEDA DE PARES DE IMÁGENES
# ============================================================

def find_pairs(img_dir: Path):
    """
    Encuentra pares de imágenes correspondientes a una
    misma secuencia representada mediante FCGR con
    512 aa y 300 aa.
    """

    all_imgs = [

        f for f in os.listdir(img_dir)

        if f.endswith(EXT)

    ]

    pairs = []

    for img in all_imgs:

        if "_512_" in img:

            base = img.replace(
                "_512_",
                "_300_"
            )

            if base in all_imgs:

                pairs.append(
                    (img, base)
                )

    return pairs


# ============================================================
# VISUALIZACIÓN DE COMPARACIONES
# ============================================================

def plot_comparison(
    pairs,
    img_dir: Path,
    n: int = 6,
    save_path="data/fcgr_images/fcgr_comparison.png"
):
    """
    Genera una cuadrícula de comparaciones
    entre imágenes FCGR de 512 y 300 aa.
    """

    if not pairs:

        print(
            "No se encontraron pares "
            "de imágenes 512/300."
        )

        return

    random.shuffle(pairs)

    pairs = pairs[:n]

    fig, axes = plt.subplots(
        n,
        2,
        figsize=(6, 3 * n)
    )

    fig.suptitle(
        "Comparativa de imágenes FCGR: "
        "512 vs 300 aminoácidos",
        fontsize=14,
        y=0.93
    )

    for i, (img512, img300) in enumerate(pairs):

        im1 = Image.open(
            img_dir / img512
        )

        im2 = Image.open(
            img_dir / img300
        )

        # ----------------------------------------------------
        # Imagen 512 aa
        # ----------------------------------------------------

        axes[i, 0].imshow(
            im1,
            cmap="gray"
        )

        axes[i, 0].set_title(
            f"{img512.split('_')[0]} (512 aa)"
        )

        axes[i, 0].axis("off")

        # ----------------------------------------------------
        # Imagen 300 aa
        # ----------------------------------------------------

        axes[i, 1].imshow(
            im2,
            cmap="gray"
        )

        axes[i, 1].set_title(
            f"{img300.split('_')[0]} (300 aa)"
        )

        axes[i, 1].axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Figura guardada en: {save_path}"
    )


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":

    pairs = find_pairs(IMG_DIR)

    print(
        f"Se encontraron {len(pairs)} "
        f"pares de imágenes FCGR "
        f"(512 vs 300)."
    )

    plot_comparison(
        pairs,
        IMG_DIR,
        n=N
    )
