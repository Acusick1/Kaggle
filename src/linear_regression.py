import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from labellines import labelLines


def np_fit():

    # Creates 50 random x and y numbers
    np.random.seed(1)
    n = 50
    x = np.random.randn(n)
    y = x * np.random.randn(n)

    # Makes the dots colorful
    colors = np.random.rand(n)

    # Plots best-fit line via polyfit
    for deg in range(1, 5):
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, deg))(np.unique(x)), label=f"n={deg}")

    labelLines(plt.gca().get_lines(), zorder=2, align=False)

    # Plots the random x and y data points we created
    # Interestingly, alpha makes it more aesthetically pleasing
    plt.scatter(x, y, c=colors, alpha=0.5)
    plt.show()


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


def linear_reg(model, inputs, labels):

    criterion = nn.MSELoss()

    epochs = 100
    learning_rate = 0.01
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):

        # Clear gradients wrt parameters
        optimiser.zero_grad()

        # Forward to get output
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Get gradients wrt parameters
        loss.backward()

        # Update parameters
        optimiser.step()

        print(f"epoch {epoch + 1}, loss {loss.item()}")

    # Get predictions
    predicted = model(inputs).cpu().data.numpy()

    # Plot true data
    plt.plot(inputs.cpu().detach(), labels.cpu(), 'go', label='True data', alpha=0.5)

    # Plot predictions
    plt.plot(inputs.cpu().detach(), predicted, '--', label='Predictions', alpha=0.5)

    # Legend and plot
    plt.legend(loc='best')
    plt.show()


def run_linear_reg(device):

    x_values = torch.arange(11)
    x_train = np.array(x_values)

    # Convert to 2D
    x_train = x_train.reshape(-1, 1)

    # Create dependent y values
    y_values = [2 * i + 1 for i in x_values]
    y_train = np.array(y_values).reshape(-1, 1)

    model = LinearRegressionModel(1, 1)
    inputs = torch.from_numpy(x_train).float().requires_grad_()
    labels = torch.from_numpy(y_train).float()

    model.to(device)
    inputs = inputs.to(device)
    labels = labels.to(device)

    linear_reg(model, inputs, labels)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_linear_reg(device)


if __name__ == "__main__":

    main()
