# ============================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================

import torch
from PIL import Image
from torch import nn, save, load
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
LEARNING_RATE = 0.001
N_EPOCHS = 10


# ============================================================
# TRANSFORMACIONES DE IMAGEN
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# ============================================================
# RUTAS DE LOS DATOS
# ============================================================

TRAIN_DIR = "../data/fcgr_512_by_classes/train"
VAL_DIR = "../data/fcgr_512_by_classes/val"
TEST_DIR = "../data/fcgr_512_by_classes/test"


# ============================================================
# DATASETS
# ============================================================

train_dataset = ImageFolder(
    root=TRAIN_DIR,
    transform=transform
)

val_dataset = ImageFolder(
    root=VAL_DIR,
    transform=transform
)

test_dataset = ImageFolder(
    root=TEST_DIR,
    transform=transform
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# DEFINICIÓN DE LA CNN
# ============================================================

class ImageClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # Capas convolucionales
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

        # Capas fully connected
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CONV4_FILTERS * 32 * 32, HIDDEN_NEURONS),
            nn.ReLU(),
            nn.Linear(HIDDEN_NEURONS, N_CLASSES)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ============================================================
# INICIALIZACIÓN DEL MODELO
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

classifier = ImageClassifier().to(device)


# ============================================================
# OPTIMIZADOR Y FUNCIÓN DE PÉRDIDA
# ============================================================

optimizer = Adam(
    classifier.parameters(),
    lr=LEARNING_RATE
)

loss_fn = nn.CrossEntropyLoss()


# ============================================================
# ENTRENAMIENTO
# ============================================================

for epoch in range(N_EPOCHS):

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = classifier(images)

        loss = loss_fn(outputs, labels)

        loss.backward()

        optimizer.step()

    print(f"Epoch: {epoch} | Loss: {loss.item()}")


# ============================================================
# GUARDAR MODELO
# ============================================================

torch.save(
    classifier.state_dict(),
    "model_state.pt"
)


# ============================================================
# CARGAR MODELO
# ============================================================

with open("model_state.pt", "rb") as f:
    classifier.load_state_dict(load(f))


# ============================================================
# INFERENCIA SOBRE UNA IMAGEN
# ============================================================

img = Image.open("image.jpg").convert("RGB")

img_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

img_tensor = img_transform(img).unsqueeze(0).to(device)

output = classifier(img_tensor)

predicted_label = torch.argmax(output)

print(f"Predicted label: {predicted_label}")
