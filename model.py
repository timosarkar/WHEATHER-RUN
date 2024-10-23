import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from safetensors.torch import save_file, load_file

epochs = 10000 
# Step 1: Create a simple dataset
# Features: [temperature, is_raining]
# Labels: 1 (can run), 0 (cannot run)
data = np.array([
    [30, 0, 1],  # 30°C, no rain -> can run
    [22, 1, 0],  # 22°C, raining -> cannot run
    [25, 0, 1],  # 25°C, no rain -> can run
    [15, 1, 0],  # 15°C, raining -> cannot run
    [20, 0, 1],  # 20°C, no rain -> can run
])

# Normalize the features
X = (data[:, :2] - np.mean(data[:, :2], axis=0)) / np.std(data[:, :2], axis=0)
y = torch.tensor(data[:, 2], dtype=torch.float32).view(-1, 1)  # Labels

# Step 2: Create a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 8)  # Increased number of neurons
        self.fc2 = nn.Linear(8, 4)   # Added an additional hidden layer
        self.fc3 = nn.Linear(4, 1)   # Hidden layer to output layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        x = self.fc1(x)
        x = nn.ReLU()(x)  # ReLU activation function
        x = self.fc2(x)
        x = nn.ReLU()(x)
        x = self.fc3(x)
        return self.sigmoid(x)  # Output

# Step 3: Train the model
model = SimpleNN()
criterion = nn.BCELoss()  # Binary cross-entropy loss
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Using Adam optimizer

# Training loop
for epoch in range(epochs):  # Increased epochs
    model.train()
    optimizer.zero_grad()  # Zero gradients
    output = model(torch.tensor(X, dtype=torch.float32))  # Forward pass
    loss = criterion(output, y)  # Compute loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights

    if (epoch + 1) % 20 == 0:  # Print every 20 epochs
        print(f'Epoch [{epoch + 1}/200], Loss: {loss.item():.4f}')

# Step 4: Save the model to Safetensors format
save_file(model.state_dict(), f'WEATHER-RUN-{epochs}.safetensors')

# Step 5: Load and run the model
def load_and_run_model(model_path, input_data):
    model = SimpleNN()
    model.load_state_dict(load_file(model_path))  # Load the model state
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # No need to compute gradients
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output = model(input_tensor)
        return output.numpy()  # Return predictions as numpy array

# Testing the model with new data
test_data = [[25, 0], [18, 1], [21, 0], [19, 1]]
normalized_test_data = (np.array(test_data) - np.mean(data[:, :2], axis=0)) / np.std(data[:, :2], axis=0)
predictions = load_and_run_model(f'WEATHER-RUN-{epochs}.safetensors', normalized_test_data)

# Output predictions
for (temp, rain), pred in zip(test_data, predictions):
    result = "can" if pred[0] >= 0.5 else "cannot"
    print(f"With temperature {temp}°C and {'rain' if rain else 'no rain'}, you {result} go for a run.")

