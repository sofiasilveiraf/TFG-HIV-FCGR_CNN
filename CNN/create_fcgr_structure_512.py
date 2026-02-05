import os
import shutil
import pandas as pd
# -----------------------------
# CONFIGURACIÓN
# -----------------------------
SPLIT_DIR = "../data/fcgr_512_split"           # carpeta con train.csv / val.csv / test.csv
IMG_COL = "png_path"                           # columna con la ruta de la imagen
LABEL_COL = "label"                            # columna con la etiqueta original
OUT_DIR = "../data/fcgr_512_by_classes"        # nueva estructura estilo PyTorch
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# LEER CSVs
# -----------------------------
train_df = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))

# -----------------------------
# NORMALIZAR RUTAS (desde src/)
# -----------------------------
def normalize_path(p):
   p = str(p)
   if os.path.exists(p):
       return p
   # si el CSV tiene rutas tipo "data/fcgr_images/xxxx.png"
   p2 = os.path.join("..", p)
   return p2

train_df["filepath"] = train_df[IMG_COL].map(normalize_path)
val_df["filepath"]   = val_df[IMG_COL].map(normalize_path)
test_df["filepath"]  = test_df[IMG_COL].map(normalize_path)

# -----------------------------
# MAPEAR ETIQUETAS A 0 y 1
# -----------------------------
# detecta automáticamente las etiquetas únicas
unique_labels = sorted(
   pd.concat([
       train_df[LABEL_COL].astype(str),
       val_df[LABEL_COL].astype(str),
       test_df[LABEL_COL].astype(str)
   ]).unique()
)
# crea diccionario ordenado para garantizar consistencia
label2idx = {lab: i for i, lab in enumerate(unique_labels)}
print("\nEtiquetas mapeadas a índices:")
print(label2idx, "\n")
train_df["label_idx"] = train_df[LABEL_COL].astype(str).map(label2idx)
val_df["label_idx"]   = val_df[LABEL_COL].astype(str).map(label2idx)
test_df["label_idx"]  = test_df[LABEL_COL].astype(str).map(label2idx)

# -----------------------------
# CREAR DIRECTORIOS /0 /1
# -----------------------------
for split, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
   split_dir = os.path.join(OUT_DIR, split)
   os.makedirs(split_dir, exist_ok=True)
   # crear carpetas de clase (0 y 1)
   for i in range(len(unique_labels)):
       os.makedirs(os.path.join(split_dir, str(i)), exist_ok=True)
   # copiar imágenes
   for _, row in df.iterrows():
       src = row["filepath"]
       cls = row["label_idx"]               # 0 o 1
       dst = os.path.join(split_dir, str(cls), os.path.basename(src))
       if not os.path.exists(src):
           print(f"⚠️  WARNING: Imagen no encontrada: {src}")
           continue
       # copia solo si no existe
       if not os.path.exists(dst):
           shutil.copy(src, dst)
   print(f"✓ Split '{split}' copiado en {split_dir}")

print("\n🎉 ESTRUCTURA COMPLETADA CORRECTAMENTE EN:")
print(OUT_DIR)