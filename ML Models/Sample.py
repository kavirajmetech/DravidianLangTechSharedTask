import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to measure training time
def train_model(device, batch_size=1000, epochs=50):
    model = SimpleNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Generate dummy data
    inputs = torch.randn(batch_size, 784, device=device)
    labels = torch.randint(0, 10, (batch_size,), device=device)
    
    start_time = time.time()
    
    for epoch in range(epochs):  # Train for more epochs
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    elapsed_time = time.time() - start_time
    return elapsed_time

# Check if CUDA is available and measure time
if torch.cuda.is_available():
    cuda_time = train_model('cuda', batch_size=10000)
    print(f"Training time on GPU (CUDA): {cuda_time:.4f} seconds")
else:
    print("CUDA is not available!")

cpu_time = train_model('cpu', batch_size=10000)
print(f"Training time on CPU: {cpu_time:.4f} seconds")
