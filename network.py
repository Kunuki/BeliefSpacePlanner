import torch as t
from torch import nn
from torchsummary import summary
from Github_Prep.data_loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def center_crop(layer, max_height, max_width):
    """Returns the crop centered around the center of given array(s) of specified size"""
    # Get the height and with of the input
    # Ignore batch size and number of channels
    _, _, h, w = layer.size()
    
    # Compute upper left corner for center crop
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    
    # return croped data
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

class UNet_TransitionDown(nn.Module):
    """Transition down block for the UNet architecture"""
    def __init__(self, In_Channel, kernel_size, device):
        super().__init__()
        # Block consists of convolution and max pool
        self.Conv = nn.Conv2d(In_Channel, 2*In_Channel, kernel_size = kernel_size, stride = 1, padding = max(int((kernel_size - 1)/2), 1))
        self.MP = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Keeps track of the device used
        self.device = device
        
        # activation function used in the block
        self.act = nn.ReLU()
        
        # Cast weights to double and move block to the specified device
        self.double()
        self.to(self.device)
        
    def forward(self, X):
        """Forward call of the block"""
        Inp = X.double().to(self.device)
        Out = self.MP(self.act(self.Conv(Inp)))
        return Out
    
class UNet_TransitionUp(nn.Module):
    """Transition Up block for the UNet archetecture"""
    def __init__(self, in_channel, append_channel, scale_factor, kernel_size, device):
        super().__init__()
        
        # Note the device the block will be running un
        self.device = device
        
        # The kernel size for the Convolutions is specified as an input parameter
        self.kernel = kernel_size
        
        # Define layers needed for the block
        self.upsample = nn.UpsamplingBilinear2d(scale_factor = scale_factor)
        self.Conv1 = nn.Conv2d(in_channel, in_channel, kernel_size = kernel_size, stride = 1, padding = max(int((kernel_size - 1)/2), 1) + 1)
        self.Conv2 = nn.Conv2d(in_channel+append_channel, int((in_channel+append_channel)/2), kernel_size = kernel_size, stride = 1, padding = max(int((kernel_size - 1)/2), 1))
        
        # Define activation function used for the block
        self.act = nn.ReLU()
        
        # Cast weights to double and move block to the specified device
        self.double()
        self.to(self.device)
        
    def forward(self, X, AppendData):
        """Forward call for an input X. Append data is the input of the skip connection"""
        # Get the size of the skip connection data and center crop to match this size
        _, _, h, w = AppendData.size()
        return self.act(self.Conv2(t.cat([center_crop(self.act(self.Conv1(self.upsample(X))), h, w), AppendData], -3)))
    
class UNet(nn.Module):
    """Neural network witn UNet architecture"""
    def __init__(self, in_channels, out_channels, kernel_size, depth, device):
        super().__init__()
        # note the input parameters
        self.device = device
        self.depth = depth
        self.kernel = kernel_size
        
        # final activation of the output is sigmoid to make the output have values in [0,1]
        self.out_act = nn.Sigmoid()
        
        # define the needed blocks depending on the depth parameter
        self.Down = nn.ModuleList([])
        self.NChannels = [in_channels]
        for n in range(depth):
            self.Down.append(UNet_TransitionDown(2**n*in_channels, kernel_size, self.device))
            self.NChannels.append(2**(n+1)*in_channels)
        
        # bottle neck layer is applied at the lowest part 
        self.bottleneck = nn.Conv2d(2**depth*in_channels, 2**depth*in_channels, kernel_size = kernel_size, stride = 1, padding = int((kernel_size - 1)/2))
        self.NChannels.append(2**depth*in_channels)
        
        # define up transition block according to the depth parameter
        self.Up = nn.ModuleList([])
        current_channels = 2**self.depth*in_channels
        for n in range(self.depth+1):
            self.Up.append(UNet_TransitionUp(current_channels, self.NChannels[-2-n], 2, kernel_size, self.device))
            current_channels = int((current_channels + self.NChannels[-2-n])/2)
            
        # At the and the final convolution is applied
        self.final_conv = nn.Conv2d(current_channels, out_channels, kernel_size = kernel_size, stride = 1, padding = int((kernel_size - 1)/2))
        
        # Cast weights to double and move block to the specified device
        self.double()
        self.to(device)
        
    
    def forward(self, X):
        X = X.double().to(self.device)
        
        ###################
        # Transition Down #
        ###################
        
        size = list()
        skip_connections = []
        skip_connections.append(X)
        for i in range(len(self.Down)):
            X = self.Down[i](X)
            skip_connections.append(X)
            size.append((X.size(-3), X.size(-2), X.size(-1)))
        
        ##############
        # Bottleneck #
        ##############
        
        X = self.bottleneck(X)
        
        #################
        # Transition Up #
        #################
        
        for i in range(self.depth+1):
            X = self.Up[i](X, skip_connections[-1-i])
            
        #####################
        # Final Convolution #
        #####################
        
        X = self.out_act(self.final_conv(X))
        
        return t.flatten(X, start_dim = 1, end_dim = 2)
    
    def save(self, PATH):
        """Function to save current weights to a weightfile with specified path"""
        t.save(self.state_dict(), PATH)
        
    def load(self, PATH):
        """load weights from a weightfile to the network"""
        self.load_state_dict(t.load(PATH, map_location=self.device), strict=False)
        
if __name__ == "__main__":
    # if a GPU is available use it. Otherwise perform operations on CPU
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    
    # define the folder where results will be saved
    SaveFolder = "0_UNet_Results_4"
    
    # create folders for results if they do not exist yet
    os.makedirs(SaveFolder+"/Weights", exist_ok = True)
    os.makedirs(SaveFolder+"/Images", exist_ok = True)
    
    # define learning rate for network training
    lr = 0.00001
    
    # initialize the neural network
    N = UNet(3, 1, 3, 5, device)
    
    # load a weightfile from which the training should start
    # if training from scratch this line has to be skipped
    N.load("0_UNet_Results_3/Weights/995.pt")
    
    # Initialize optimization algorithm on network parameters
    Trainer = t.optim.Adam(N.parameters(), lr = lr)
    
    # Define loss function for training
    LossFct = nn.BCELoss()
    
    
    # Create lists for the log of the loss function throughout training
    Loss_Log = list()
    Loss_Log_Test = list()
    
    # Initialize data loader to load training data
    D = DataLoader(32, "FormattedData")
        
    # Load test data
    X_Test, Y_Test = D.get_test()

    # Here the number of epochs the training is run for can be specified
    for epoch in tqdm(range(100), desc = "Epoch"):
        
        # Initialize total loss values to keep track of loss thoughout epoch
        Total_Loss = 0
        Total_N = 0
        
        # iterate over batches of the training data using the data loader
        for X, Y in D:
            
            # Get the estimate for the input batch
            f = N.forward(X.to(device))
            
            # Compute loss function for the batch
            L = LossFct(f, Y.to(device))
            
            # Set parameter gradients to zero
            Trainer.zero_grad()
            
            # Compute parameter gradients from the loss
            L.backward()
            
            # perform optimization step using the computed gradients
            Trainer.step()
            
            # add the loss value of the batch to the total loss
            Total_Loss += X.size(0)*L.item()
            
            # also keep track of the total number of data points throughout all batches
            Total_N += X.size(0)

        # Store the relative training loss of epoch to the list
        Loss_Log.append(Total_Loss/Total_N)
        
        # Evalutate the test loss for the current weights
        with t.no_grad():
            Estimate = N.forward(X_Test.to(device))
            L = LossFct(Estimate, Y_Test.to(device))
            Loss_Log_Test.append(L.item())
            Estimate = Estimate.cpu().numpy()
            Y_Test_Array = Y_Test.cpu().numpy()

        # Every 5 epochs
        if epoch % 5 == 0:
            # plot the logged training and test loss values
            # this is used to keep track of training process
            plt.clf()
            plt.plot(Loss_Log, label = "Train Loss")
            plt.plot(Loss_Log_Test, label = "Test Loss")
            plt.legend()
            plt.savefig(SaveFolder+"/Training.png")
            
            # plot estimates for manual inspection
            for i in range(Estimate.shape[0]):
                plt.clf()
                plt.subplot(1, 2, 1)
                plt.contourf(Estimate[i,:,:])
                plt.title("Estimate")
                plt.subplot(1, 2, 2)
                plt.contourf(Y_Test_Array[i,:,:])
                plt.title("Optimal Path")
                plt.savefig(SaveFolder+"/Images/Test_"+str(i+1)+".png")
            
            # Save the current weights to a weightfile
            N.save(SaveFolder+"/Weights/"+str(epoch)+".pt")
                
        
    
        
        
        
            
            