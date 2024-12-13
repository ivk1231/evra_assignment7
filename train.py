import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from model import Model_1, Model_2, Model_3

# Define data augmentation
transform = transforms.Compose([
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Function to train a model
def train_model(model_class, model_name):
    model = model_class()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(15):
        model.train()
        total_loss = 0
        correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(train_loader.dataset)
        print(f'{model_name} - Epoch {epoch+1}: Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if accuracy >= 99.4:
            print(f"{model_name} - Target accuracy reached.")
            break

        scheduler.step()

    with open(f"{model_name}_accuracy.txt", "w") as f:
        f.write(f"Final Accuracy: {accuracy:.2f}%\n")

if __name__ == "__main__":
    # Train each model and print results
    print("Training Model_1")
    train_model(Model_1, "Model_1")
    print("Training Model_2")
    train_model(Model_2, "Model_2")
    print("Training Model_3")
    train_model(Model_3, "Model_3")