# Importing dependencies
import torch
from PIL import Image
from torch import nn,save,load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Loading Data (simple, usando ImageFolder y la estructura 0/1)
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# 1. Transformaciones básicas
transform = transforms.Compose([
   transforms.Resize((512, 512)),   # ajustamos a 512x512
   transforms.ToTensor()
])
# 2. Rutas a tus carpetas ya creadas
TRAIN_DIR = "../data/fcgr_512_by_classes/train"
VAL_DIR   = "../data/fcgr_512_by_classes/val"
TEST_DIR  = "../data/fcgr_512_by_classes/test"
# 3. Crear datasets
train_dataset = ImageFolder(root=TRAIN_DIR, transform=transform)
val_dataset   = ImageFolder(root=VAL_DIR,   transform=transform)
test_dataset  = ImageFolder(root=TEST_DIR,  transform=transform)
# 4. Dataloaders para la CNN
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)


# Define the image classifier model
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 512 → 256
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 256 → 128
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 128 → 64
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)    # 64 → 32
        )

        # 128 filtros, 32x32 tamaño final
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # salida = 2 clases (0 y 1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Create an instance of the image classifier model 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = ImageClassifier().to(device)


# Define the optimizer and loss function
optimizer = Adam(classifier.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Train the model
for epoch in range(10):  # Train for 10 epochs
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  # Reset gradients
        outputs = classifier(images)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    print(f"Epoch:{epoch} loss is {loss.item()}")


# Save the trained model
torch.save(classifier.state_dict(), 'model_state.pt')


# Load the saved model
with open('model_state.pt', 'rb') as f: 
     classifier.load_state_dict(load(f))  
       

# Perform inference on an image
from PIL import Image
img = Image.open("image.jpg").convert("RGB")
img_transform = transforms.Compose([
   transforms.Resize((512, 512)),
   transforms.ToTensor()
])
img_tensor = img_transform(img).unsqueeze(0).to(device)
output = classifier(img_tensor)
predicted_label = torch.argmax(output)
print(f"Predicted label: {predicted_label}")


