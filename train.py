import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from model import Model_1, Model_2, Model_3

# Define data augmentation with reduced rotation
transform = transforms.Compose([
    transforms.RandomRotation((-3.0, 3.0)),  # Reduced rotation angle
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load dataset
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)

def train_model(model_class, model_name, accuracy_file):
    model = model_class()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_accuracy = 0
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

        # Update learning rate based on accuracy
        scheduler.step(accuracy)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy

        if accuracy >= 99.4:
            print(f"{model_name} - Target accuracy reached.")
            break

    with open(accuracy_file, "a") as f:
        f.write(f"{model_name} - Final Accuracy: {accuracy:.2f}%, Best Accuracy: {best_accuracy:.2f}%\n")

if __name__ == "__main__":
    # Clear the accuracy file
    with open("accuracy.txt", "w") as f:
        f.write("Training Results:\n")

    print("\nTraining Model_1:")
    train_model(Model_1, "Model_1", "accuracy.txt")
    print("\nTraining Model_2:")
    train_model(Model_2, "Model_2", "accuracy.txt")
    print("\nTraining Model_3:")
    train_model(Model_3, "Model_3", "accuracy.txt")