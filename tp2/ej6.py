# %% [markdown]
# Entrene una red convolucional para clasificar las imágenes de la base de datos MNIST.
# 
# ¿Cuál es la red convolucional más pequeña que puede conseguir con una exactitud de al menos 90% en el conjunto de evaluación? ¿Cuál es el perceptrón multicapa más pequeño que puede conseguir con la misma exactitud?

# %%
import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

# %% [markdown]
# En este caso se nos permitió usar Pytorch para la parte de código. La idea es definir la estructura general de la red y luego ir modificandola hasta lograr la mejor performance. 

# %%
# Define a transform to convert the data to tensor
transform = transforms.ToTensor()

# Load the training dataset
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)

# Load the test dataset
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# %% [markdown]
# La primer estructura que quiero hacer es algo muy simple, una red convolucional pegada a una fully connected, bien simple. 

# %%
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) # son 6 filtros de 5x5, pero bienen en escala de grises
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # esto reduce la dimensión espacial a la mitad
        self.flatten = nn.Flatten() # esto aplana la entrada para la capa lineal
        self.linear_relu_stack = nn.Sequential( # solo 1 capa lineal, mínimo posible
            nn.Linear(864, 10), 
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv → ReLU → pool
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# %%
batch_size = 64  # minibatch
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# %% [markdown]
# El código de abajo entrena hasta un máximo de 20 epochs, con early stippong (si converge la loss). Entrena y testea por cada epoch. también grafica todo para poder entender como varia con epoch, la idea es que podemos comparar la velocidad. 

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def train_and_evaluate(model, train_loader, test_loader, criterion= nn.CrossEntropyLoss(), optimizer, device,
                       num_epochs=100, patience=10, min_delta=1e-4, plot=True):
    """
    Entrena y evalúa un modelo PyTorch con early stopping.
    
    Args:
        model: instancia de nn.Module
        train_loader: DataLoader de entrenamiento
        test_loader: DataLoader de validación/prueba
        criterion: función de pérdida (ej: nn.CrossEntropyLoss())
        optimizer: optimizador (ej: optim.Adam(model.parameters(), lr=0.001))
        device: 'cpu' o 'cuda'
        num_epochs: cantidad máxima de épocas
        patience: cuántas épocas esperar sin mejora
        min_delta: mejora mínima en test loss para resetear early stopping
        plot: si True, grafica loss y accuracy

    Returns:
        history: diccionario con 'train_loss', 'test_loss', 'accuracy'
    """

    model = model.to(device)
    best_loss = float('inf')
    epochs_no_improve = 0
    print(model)


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"El modelo tiene {total_params:,} parámetros entrenables.")

    history = {
        "train_loss": [],
        "test_loss": [],
        "accuracy": []
    }

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)

        # --- Evaluación ---
        model.eval()
        test_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_test_loss = test_loss / len(test_loader)
        acc = 100 * correct / total

        history["test_loss"].append(avg_test_loss)
        history["accuracy"].append(acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | "
              f"Acc: {acc:.2f}%")

        # --- Early stopping ---
        if avg_test_loss < best_loss - min_delta:
            best_loss = avg_test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print("Early stopping por falta de mejora.")
            break

    # --- Graficar ---
    if plot:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["test_loss"], label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Evolución del Loss")

        plt.subplot(1, 2, 2)
        plt.plot(history["accuracy"], label="Accuracy", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Evolución de la Accuracy")
        plt.tight_layout()
        plt.show()

    return history


# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

history = train_and_evaluate(
    model, 
    train_loader, 
    test_loader, 
    criterion, 
    optimizer, 
    device,
    num_epochs=50
)



# %% [markdown]
# ahora creo otros modelos y vemos que pasa. la idea es ir mejorando para llegar a la performance de 90%. primero un modelo con más capa de perceptron. 

# %%
# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) # son 6 filtros de 5x5, duplicamos la cantidad de kernels diferentes. -> 26x26x6
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # esto reduce la dimensión espacial a la mitad -> 12x12x6
        self.flatten = nn.Flatten() # esto aplana la entrada para la capa lineal
        # After conv (28->24) and pool (24->12), the feature map size is 12x12 with 6 channels -> 12*12*6 = 864
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(12 * 12 * 6, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv → ReLU → pool
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)


# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

history = train_and_evaluate(
    model, 
    train_loader, 
    test_loader, 
    criterion, 
    optimizer, 
    device,
    num_epochs=50
)



# %%


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5) # son 6 filtros de 5x5, pero bienen en escala de grises
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # esto reduce la dimensión espacial a la mitad
        self.flatten = nn.Flatten() # esto aplana la entrada para la capa lineal
        self.linear_relu_stack = nn.Sequential( # solo 1 capa lineal, mínimo posible
            nn.Linear(864, 10), 
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv → ReLU → pool
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# %% [markdown]
# Ahora comparamos con el perceptrón muilticapa más chico que encontremos. 


