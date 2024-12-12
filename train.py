import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from model import Model_1, Model_2

def train_model(model_class):
    # Data loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Model, loss, optimizer
    model = model_class()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
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
        print(f'Epoch {epoch+1}: Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%')

        if accuracy >= 99.4:
            print("Target accuracy reached.")
            break

if __name__ == "__main__":
    # Choose which model to train
    train_model(Model_1)  # or train_model(Model_2) 