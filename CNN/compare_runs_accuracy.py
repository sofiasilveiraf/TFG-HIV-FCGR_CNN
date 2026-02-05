# compare_runs_accuracy.py
# Ejecutar desde TFG_Sofia2/src
# python3 compare_runs_accuracy.py
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
# -------------------------
# Config (ajusta aquí)
# -------------------------
TRAIN_DIR = "../data/fcgr_512_by_classes/train"
VAL_DIR   = "../data/fcgr_512_by_classes/val"
TEST_DIR  = "../data/fcgr_512_by_classes/test"
batch_size = 32   # reduce a 16 o 8 si te falta memoria
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
# Experiment grid (ajusta a lo que quieras comparar)
n_epochs_list = [10, 20, 30]           # ejemplo: 10, 20, 30, (puedes añadir 100)
learning_rates = [1e-3, 5e-4]          # prueba varios LR
# Output CSV para resultados
OUT_CSV = "results_by_run.csv"
# -------------------------
# Data loaders
# -------------------------
transform = transforms.Compose([
   transforms.Resize((512, 512)),
   transforms.ToTensor()
])
train_ds = ImageFolder(root=TRAIN_DIR, transform=transform)
val_ds   = ImageFolder(root=VAL_DIR,   transform=transform)
test_ds  = ImageFolder(root=TEST_DIR,  transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
print("Clases detectadas:", train_ds.class_to_idx)
print("N train, val:", len(train_ds), len(val_ds))
# -------------------------
# Modelo sencillo (igual que usamos)
# -------------------------
class ImageClassifier(nn.Module):
   def __init__(self):
       super().__init__()
       self.conv_layers = nn.Sequential(
           nn.Conv2d(3, 16, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.Conv2d(16, 32, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.Conv2d(32, 64, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2),
           nn.Conv2d(64, 128, kernel_size=3, padding=1),
           nn.ReLU(),
           nn.MaxPool2d(2)
       )
       self.fc = nn.Sequential(
           nn.Flatten(),
           nn.Linear(128 * 32 * 32, 64),
           nn.ReLU(),
           nn.Linear(64, 2)
       )
   def forward(self, x):
       x = self.conv_layers(x)
       x = self.fc(x)
       return x
# -------------------------
# Helpers: evaluación por clases
# -------------------------
def evaluate_per_class(model, loader, device):
   model.eval()
   correct_total = 0
   total = 0
   # asumimos clases 0 y 1
   per_class_correct = {0: 0, 1: 0}
   per_class_total   = {0: 0, 1: 0}
   with torch.no_grad():
       for images, labels in loader:
           images = images.to(device)
           labels = labels.to(device)
           outputs = model(images)
           preds = outputs.argmax(dim=1)
           correct_total += (preds == labels).sum().item()
           total += labels.size(0)
           for c in [0,1]:
               mask = (labels == c)
               if mask.any():
                   per_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                   per_class_total[c]   += int(mask.sum().item())
   overall_acc = correct_total / total if total>0 else 0.0
   per_class_acc = {}
   for c in [0,1]:
       per_class_acc[c] = per_class_correct[c] / per_class_total[c] if per_class_total[c] > 0 else None
   return overall_acc, per_class_acc
# -------------------------
# Loop de experimentos
# -------------------------
results = []  # listas de dict para guardar en DataFrame
run_id = 0
for lr in learning_rates:
   for n_epochs in n_epochs_list:
       run_id += 1
       print("\n" + "="*60)
       print(f"RUN {run_id} -> lr={lr}, epochs={n_epochs}")
       print("="*60)
       # nuevo modelo por cada run
       model = ImageClassifier().to(device)
       optimizer = Adam(model.parameters(), lr=lr)
       loss_fn = nn.CrossEntropyLoss()
       # Entrenamiento por epochs
       for epoch in range(1, n_epochs+1):
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
           # Evaluación en train y val (en modo eval, no grad)
           train_acc, train_per_class = evaluate_per_class(model, train_loader, device)
           val_acc,   val_per_class   = evaluate_per_class(model, val_loader, device)
           # Imprime resumen de epoch
           print(f"Run {run_id} | lr {lr} | E{epoch}/{n_epochs} | loss {epoch_loss:.4f} | "
                 f"train_acc {train_acc:.4f} (c0 {train_per_class[0]}, c1 {train_per_class[1]}) | "
                 f"val_acc {val_acc:.4f} (c0 {val_per_class[0]}, c1 {val_per_class[1]})")
           # Guardar fila en resultados
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
       # fin run
       # opcional: liberar memoria GPU
       del model
       torch.cuda.empty_cache()
# -------------------------
# Guardar CSV con todos los resultados
# -------------------------
df = pd.DataFrame(results)
df.to_csv(OUT_CSV, index=False)
print("\nResultados guardados en", OUT_CSV)