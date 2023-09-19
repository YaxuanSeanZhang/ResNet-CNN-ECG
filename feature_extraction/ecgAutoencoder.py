import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Define the autoencoder model
class ECGAutoencoder(nn.Module):
    def __init__(self, input_shape):
        """
        :param input_shape:
        """

        super(ECGAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(12*1000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )


        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 12*1000),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, train_loader, num_epochs, learning_rate):
        """
        Train the autoencoder model.
        For ECG data, the input shape is [batch_size, 12, 5000]
        The optimizer is Adam
        :param train_loader: Training data loader
        :param num_epochs: Number of epochs
        :param learning_rate: Learning rate
        :param batch_size:
        :return:
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # Train the model
        loss_list = []
        for epoch in range(num_epochs):
            # Track cumulative loss
            cumulative_loss = 0
            for i, data in enumerate(train_loader):
                # print(data.shape)
                # Get the inputs
                inputs = data[0].view(-1, 12*1000)
                # print(inputs.shape)
                # inputs = data
                optimizer.zero_grad()
                # Forward pass
                outputs = self(inputs)
                # Convert outputs to the same data type as inputs
                # outputs = outputs[1].type_as(inputs)
                # Compute the loss
                loss = criterion(outputs, inputs)
                # Backward pass
                # optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Track cumulative loss
                cumulative_loss += loss.item()
            loss_list.append(cumulative_loss)
            print("Epoch: %d, Cumulative Loss: %f" % (epoch + 1, cumulative_loss))

            # Plot the cumulative loss at the end of each epoch
            # range of epochs
            epoch_range = range(1, epoch + 1)
            # Plot the loss curve
            plt.plot(loss_list)
            plt.xlabel('Epoch')
            plt.ylabel('Cumulative Loss')
            plt.title('Autoencoder: Training Loss')
            plt.xticks(epoch_range)
            plt.savefig("autoencoder_loss.png")

            self.train_loss = loss_list




