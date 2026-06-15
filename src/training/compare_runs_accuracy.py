# ============================================================
# COMPARE_RUNS_ACCURACY.PY
# ============================================================
#
# Ejecutar desde:
#
#   TFG_Sofia2/src
#
# Comando:
#
#   python3 compare_runs_accuracy.py
#
# ============================================================


# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import os
import copy
import time

import pandas as pd

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder


# ============================================================
# PARÁMETROS DEL PROYECTO
# ============================================================

IMAGE_SIZE = 512

INPUT_CHANNELS = 3

CONV1_FILTERS = 16
CONV2_FILTERS = 32
CONV3_FILTERS = 64
CONV4_FILTERS = 128

KERNEL_SIZE = 3
PADDING = 1
POOL_SIZE = 2

HIDDEN_NEURONS = 64
N_CLASSES = 2

BATCH_SIZE = 32
NUM_WORKERS = 2

LEARNING_RATES = [1e-3, 5e-4]
N_EPOCHS_LIST = [10, 20, 30]


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

TRAIN_DIR = "../data/fcgr_512_by_classes/train"
VAL_DIR   = "../data/fcgr_512_by_classes/val"
TEST_DIR  = "../data/fcgr_512_by_classes/test"

batch_size = BATCH_SIZE

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# CONFIGURACIÓN DE EXPERIMENTOS
# ============================================================

n_epochs_list = N_EPOCHS_LIST
learning_rates = LEARNING_RATES

OUT_CSV = "results_by_run.csv"


# ============================================================
# TRANSFORMACIONES DE IMAGEN
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# ============================================================
# DATASETS
# ============================================================

train_ds = ImageFolder(
    root=TRAIN_DIR,
    transform=transform
)

val_ds = ImageFolder(
    root=VAL_DIR,
    transform=transform
)

test_ds = ImageFolder(
    root=TEST_DIR,
    transform=transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print("Clases detectadas:", train_ds.class_to_idx)
print("N train, val:", len(train_ds), len(val_ds))


# ============================================================
# MODELO CNN
# ============================================================

class ImageClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv_layers = nn.Sequential(

            nn.Conv2d(
                INPUT_CHANNELS,
                CONV1_FILTERS,
                kernel_size=KERNEL_SIZE,
                padding=PADDING
            ),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),

            nn.Conv2d(
                CONV1_FILTERS,
                CONV2_FILTERS,
                kernel_size=KERNEL_SIZE,
                padding=PADDING
            ),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),

            nn.Conv2d(
                CONV2_FILTERS,
                CONV3_FILTERS,
                kernel_size=KERNEL_SIZE,
                padding=PADDING
            ),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE),

            nn.Conv2d(
                CONV3_FILTERS,
                CONV4_FILTERS,
                kernel_size=KERNEL_SIZE,
                padding=PADDING
            ),
            nn.ReLU(),
            nn.MaxPool2d(POOL_SIZE)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                CONV4_FILTERS * 32 * 32,
                HIDDEN_NEURONS
            ),
            nn.ReLU(),
            nn.Linear(
                HIDDEN_NEURONS,
                N_CLASSES
            )
        )

    def forward(self, x):

        x = self.conv_layers(x)
        x = self.fc(x)

        return x


# ============================================================
# EVALUACIÓN POR CLASE
# ============================================================

def evaluate_per_class(model, loader, device):

    model.eval()

    correct_total = 0
    total = 0

    per_class_correct = {0: 0, 1: 0}
    per_class_total = {0: 0, 1: 0}

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            preds = outputs.argmax(dim=1)

            correct_total += (preds == labels).sum().item()
            total += labels.size(0)

            for c in [0, 1]:

                mask = (labels == c)

                if mask.any():

                    per_class_correct[c] += (
                        preds[mask] == labels[mask]
                    ).sum().item()

                    per_class_total[c] += int(
                        mask.sum().item()
                    )

    overall_acc = (
        correct_total / total
        if total > 0 else 0.0
    )

    per_class_acc = {}

    for c in [0, 1]:

        per_class_acc[c] = (
            per_class_correct[c] / per_class_total[c]
            if per_class_total[c] > 0
            else None
        )

    return overall_acc, per_class_acc


# ============================================================
# BUCLE DE EXPERIMENTOS
# ============================================================

results = []

run_id = 0

for lr in learning_rates:

    for n_epochs in n_epochs_list:

        run_id += 1

        print("\n" + "=" * 60)
        print(f"RUN {run_id} -> lr={lr}, epochs={n_epochs}")
        print("=" * 60)

        model = ImageClassifier().to(device)

        optimizer = Adam(
            model.parameters(),
            lr=lr
        )

        loss_fn = nn.CrossEntropyLoss()

        # ----------------------------------------------------
        # ENTRENAMIENTO
        # ----------------------------------------------------

        for epoch in range(1, n_epochs + 1):

            model.train()

            epoch_loss = 0.0
            n_samples = 0

            for images, labels in train_loader:

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                loss = loss_fn(outputs, labels)

                loss.backward()

                optimizer.step()

                epoch_loss += loss.item() * images.size(0)
                n_samples += images.size(0)

            epoch_loss = epoch_loss / max(1, n_samples)

            train_acc, train_per_class = evaluate_per_class(
                model,
                train_loader,
                device
            )

            val_acc, val_per_class = evaluate_per_class(
                model,
                val_loader,
                device
            )

            print(
                f"Run {run_id} | lr {lr} | "
                f"E{epoch}/{n_epochs} | "
                f"loss {epoch_loss:.4f} | "
                f"train_acc {train_acc:.4f} "
                f"(c0 {train_per_class[0]}, c1 {train_per_class[1]}) | "
                f"val_acc {val_acc:.4f} "
                f"(c0 {val_per_class[0]}, c1 {val_per_class[1]})"
            )

            results.append({
                "run_id": run_id,
                "lr": lr,
                "n_epochs_setting": n_epochs,
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_acc": train_acc,
                "train_acc_c0": train_per_class[0],
                "train_acc_c1": train_per_class[1],
                "val_acc": val_acc,
                "val_acc_c0": val_per_class[0],
                "val_acc_c1": val_per_class[1],
                "timestamp": time.time()
            })

        # Liberar memoria GPU

        del model

        torch.cuda.empty_cache()


# ============================================================
# GUARDAR RESULTADOS
# ============================================================

df = pd.DataFrame(results)

df.to_csv(
    OUT_CSV,
    index=False
)

print("\nResultados guardados en", OUT_CSV)
