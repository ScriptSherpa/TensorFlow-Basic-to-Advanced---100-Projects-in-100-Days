# # import torch

# # print("CUDA Available:", torch.cuda.is_available())
# # if torch.cuda.is_available():
# #     print("GPU Name:", torch.cuda.get_device_name(0))
# #     print("CUDA Version:", torch.version.cuda)
# #     print("Torch Version:", torch.__version__)
# # import torch
# # print(torch.version.cuda)  # Shows CUDA version PyTorch was compiled with
# # print(torch.backends.cudnn.enabled)  # Should be True
# # print(torch.cuda.device_count())  # Number of GPUs available


# # import torch
# # import time

# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # print(f"Running on device: {device}")

# # # Create two large random matrices
# # x = torch.randn((10000, 10000), device=device)
# # y = torch.randn((10000, 10000), device=device)

# # # Warm up GPU
# # torch.matmul(x, y)

# # # Time the matrix multiplication
# # start = time.time()
# # result = torch.matmul(x, y)
# # torch.cuda.synchronize()  # Wait for the operation to finish
# # end = time.time()

# # print(f"Matrix multiplication took {end - start:.4f} seconds on {device}")
# import torch
# import torchvision
# import torchaudio
# import requests
# import tqdm
# import matplotlib

# print("✅ Installed Package Versions:")
# print(f"Torch:         {torch.__version__}")
# print(f"Torchvision:   {torchvision.__version__}")
# print(f"Torchaudio:    {torchaudio.__version__}")
# print(f"Requests:      {requests.__version__}")
# print(f"TQDM:          {tqdm.__version__}")
# print(f"Matplotlib:    {matplotlib.__version__}")
# ------------------------------------------------------------------
import os
import zipfile
import requests
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Step 1: Download and extract dataset if not present
dataset_url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
zip_path = "cats_and_dogs_filtered.zip"
data_dir = "./cats_and_dogs_filtered"

if not os.path.exists(data_dir):
    print("Downloading dataset...")
    r = requests.get(dataset_url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in tqdm(r.iter_content(chunk_size=1024)):
            if chunk:
                f.write(chunk)
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall()

data_path = os.path.join(data_dir, "train")

# Step 2: Define transforms (resize + normalize)
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize RGB channels to [-1, 1]
])

# Step 3: Load dataset
full_dataset = datasets.ImageFolder(data_path, transform=transform)

# Step 4: Split dataset
val_split = 0.2
test_split = 0.1
total_size = len(full_dataset)
val_size = int(val_split * total_size)
test_size = int(test_split * total_size)
train_size = total_size - val_size - test_size

train_ds, val_ds, test_ds = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

# Step 5: Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

# Step 6: Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20 * 20, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Step 7: Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("Is CUDA available:", torch.cuda.is_available())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


model = SimpleCNN().to(device)

# Step 8: Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
val_accuracies = []
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")

# Step 9: Training loop
epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total

    # Save metrics
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
          f"Val Acc: {val_acc:.4f}")

# Step 10: (Optional) Test accuracy
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)
test_acc = test_correct / test_total
print(f"Test Accuracy: {test_acc:.4f}")

# Step 11: Plot training curves
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Training Metrics')
plt.legend()
plt.grid(True)
plt.show()

torch.save(model.state_dict(), "simple_cnn.pth")
print("Model saved to simple_cnn.pth")
