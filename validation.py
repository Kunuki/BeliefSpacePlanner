from network_v2 import UNet
from data_loader import DataLoader
import torch as t
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from sklearn.mixture import GaussianMixture
from heapq import heapify, heappop, heappush
import time
from skfmm import distance

def get_loss_history(FolderList, DataFolder):
    """Get list of loss values (Training and Test) from a folder of weights files"""
    
    # Create lists to store loss values as well as epoch numbers (Epochs are saved in increments of 5)
    X_Values = list()
    Loss_Train = list()
    Loss_Test = list()
    
    # Initialize the data loader
    D = DataLoader(32, DataFolder)
    
    # Define the loss function used for the loss value computation
    Loss_Fct = nn.BCELoss()
    
    # Load the test data set (Training data is loaded later in batches)
    X_Test, Y_Test = D.get_test()
    
    # Initialize the network archtecture
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    N = UNet(3, 1, 3, 5, device)
    
    
    for Folder in FolderList:
        # Get index list of the epochs in the stored weight files in the folder
        Epoch_List = sorted([int(x.split(".")[0]) for x in os.listdir(Folder+"/Weights")])
        
        for epoch in tqdm(Epoch_List, desc = Folder):
            # Load the weight file into the network
            WeightFile = Folder+"/Weights/"+str(epoch)+".pt"
            N.load(WeightFile)
            
            # Store cumulative epoch number
            if len(X_Values) == 0:
                X_Values.append(0)
            else:
                X_Values.append(X_Values[-1]+5)
            
            with t.no_grad():
                # Evaluate test loss for the network with current weightfile
                Test_Est = N.forward(X_Test.to(device))
                L = Loss_Fct(Test_Est, Y_Test.to(device)).cpu().item()
                Loss_Test.append(L)
                
                # Evaluate cumulative train loss for the network with current weightfile using batches of training data
                Loss_Train_Total = 0
                N_Train_Total = 0
                for X, Y in D:
                    Train_Est = N.forward(X.to(device))
                    L = Loss_Fct(Train_Est, Y.to(device)).cpu().item()
                    Loss_Train_Total += X.size(0)*L
                    N_Train_Total += X.size(0)
                Loss_Train.append(Loss_Train_Total/N_Train_Total)
    
    # Create plot of the loss curves
    plt.clf()
    plt.plot(X_Values, Loss_Train, label = "Train Loss")
    plt.plot(X_Values, Loss_Test, label = "Test Loss")
    plt.legend()
    plt.savefig("0_Validation/Training_Process.png")

def create_gif(image_files, output_filename, duration=500, loop=0):
    # Open the images and store them in a list
    images = [Image.open(image) for image in image_files]
    
    # Save the images as a GIF
    images[0].save(
        output_filename,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=loop
    )

def moving_circle(WeightFile, DataFolder):
    # Initialize the neural network
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    N = UNet(3, 1, 3, 5, device)
    
    # Load the given weightfile
    N.load(WeightFile)
    
    # Initialize the data load
    D = DataLoader(32, DataFolder)
    
    # Load the test data set
    X_Test, Y_Test = D.get_test()
    
    # Cast the test data to numpy arrays
    X_Test = X_Test.numpy()
    Y_Test = Y_Test.numpy()
    
    for IDX in [14]:
        # Generate a random circle
        mid = np.array([np.random.rand(), 0.5*np.random.rand()])
        radius = 0.1
        
        
        for n in range(10):
            # Create a copy of the input of the test data set
            New_Inp = np.array(X_Test)

            # Add the circle to the indicator array describing the obstacle in the copied input data
            for i in range(X_Test.shape[2]):
                for j in range(X_Test.shape[3]):
                    if np.linalg.norm(np.array([i/(X_Test.shape[2]-1),j/(X_Test.shape[3]-1)]) - mid) < radius:
                        New_Inp[IDX, 0, i,j] = 1
        
            # Compute prediction of the input with added circle
            with t.no_grad():
                Prediction = N.forward(t.tensor(New_Inp[[IDX], :, :, :])).cpu().numpy()
            
            # Move the circle in y direction for the next iteration
            mid += np.array([0, 0.1])
            mid = np.clip(mid, 0, 1)
            
            # Create a plot of the current estimate for the input with added circle
            plt.clf()
            plt.contourf(Prediction[0, :, :])
            plt.contour(New_Inp[IDX, 0, :, :])
            plt.contour(1*(New_Inp[IDX, 1, :, :] < 0.001), colors = "red")
            plt.savefig("0_Validation/MovingCircles/"+str(IDX+1)+"_"+str(n+1)+".png")
        
        # process the produced plots into a gif   
        file_names = ["0_Validation/MovingCircles/"+str(IDX+1)+"_"+str(n+1)+".png" for n in range(10)]
        create_gif(file_names, "0_Validation/MovingCircles/"+str(IDX)+".gif")
        
        # delete the stored images that are not needed anymore
        for file in file_names:
            os.remove(file)


def bilinear_interpolation(x, y, Array):
    """Get bilinear interpolation of an array interpreted as values on a uniform grid on [0,1]^2"""
    
    # x and y are values between 0 and 1
    # transform them into the index range needed for the array shape
    # take the floor the get the next smaller index
    # next larger index is xl+1 and yl+1
    xl = int((Array.shape[0]-1)*x)
    yl = int((Array.shape[1]-1)*y)
    
    # evalulate value of the array on all combinations of larger and smaller indices
    # if the adjacent value is not available 0 is used instead
    A11 = Array[xl, yl]
    
    if yl+1 < Array.shape[1]:
        A12 = Array[xl, yl+1]
    else:
        A12 = 0
        
    if xl+1 < Array.shape[0]:
        A21 = Array[xl+1, yl]
    else:
        A21 = 0
        
    if xl + 1 < Array.shape[0] and yl + 1 < Array.shape[1]:
        A22 = Array[xl+1, yl+1]
    else:
        A22 = 0
    
    # get the unrounded index value as floating point numbers
    X = (Array.shape[0]-1)*x
    Y = (Array.shape[1]-1)*y
    
    # Compute bilinear interpolation of computed values
    Val = A11*(xl+1-X)*(yl+1-Y) + A12*(xl+1-X)*(Y-yl) + A21*(X-xl)*(yl+1-Y) + A22*(X-xl)*(Y-yl)
    return Val
    

def get_distance_matrix(x, y, shape):
    """Create distance matrix of given shape containing distance to given point [x,y] in [0,1]^2"""
    
    # create array of the needed shape
    A = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # store distance of the index pair mapped to the interval [0,1] to the given point [x,y] in [0,1]^2
            A[i,j] = np.linalg.norm(np.array([i/(shape[0]-1), j/(shape[1]-1)]) - np.array([x, y]))
            
    # Normalize the computed array for network training
    m = np.min(A)
    M = np.max(A)
    A = (A - m)/(M-m)
    return A
             
def store_data(WeightFile, DataFolder):
    """Store results of a trainined neural network with a given weight file for further analysis"""
    
    # Initialize the neural network
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    N = UNet(3, 1, 3, 5, device)
    
    # load the given weightfile into the network
    N.load(WeightFile)
    
    # Initialize data loader to load test data set
    D = DataLoader(32, DataFolder)
    X_Test, Y_Test = D.get_test()
    
    # Cast test data to numpy arrays
    X_Test = X_Test.numpy()
    Y_Test = Y_Test.numpy()
    
    
    for i in tqdm(range(X_Test.shape[0]), desc = "Storing Data"):
        # Extract out different parts of the input and label to store seperately
        Map = X_Test[i, 0, :, :]
        Start = X_Test[i, 1, :, :]
        Goal = X_Test[i, 2, :, :]
        Label = Y_Test[i, :, :]
        
        # Comppute the path prediction using the neural network for the given input
        with t.no_grad():
            Prediction = N.forward(t.tensor(X_Test[[i],:,:,:])).cpu().numpy()[0,:,:]
        
        # Store components of the input, label and prediction seperately
        np.savetxt("0_Validation/Data/Map_"+str(i+1)+".csv", Map)
        np.savetxt("0_Validation/Data/Start_"+str(i+1)+".csv", Start)
        np.savetxt("0_Validation/Data/Goal_"+str(i+1)+".csv", Goal)
        np.savetxt("0_Validation/Data/Label_"+str(i+1)+".csv", Label)
        np.savetxt("0_Validation/Data/Prediction_"+str(i+1)+".csv", Prediction)
        
def sample_by_image(Image, N):
    """Sample N points from 2D distribution from an Image interpreted as a density function"""
    
    # get the shape of the image
    Nx = Image.shape[0]
    Ny = Image.shape[1]
    
    # normalize given image to be interpreted as density function
    Image = Image/np.sum(Image)
    
    # create list to store sampled points
    Points = list()
    
    for _ in range(N):
        # Compute cdf for first coordinate of the density
        Marginal = np.sum(Image, axis = 0).tolist()
        Marginal = [sum(Marginal[:ii]) for ii in range(len(Marginal))]
        
        # Sample from the cdf of the first coordinate using inverse cdf method
        U1 = np.random.rand()
        i = len(Marginal) - 2
        while Marginal[i] > U1 and i > 0:
            i -= 1
        y = (U1 - Marginal[i])/(Marginal[i+1] - Marginal[i])*i/Ny + (Marginal[i+1] - U1)/(Marginal[i+1] - Marginal[i])*(i+1)/Ny
        
        # Compute the conditional cdf of the second coordinate given the sampled first coordinate
        Conditional = [Image[jj, i]/np.sum(Image[:, i]) for jj in range(Image.shape[0])]
        Conditional = [sum(Conditional[:ii]) for ii in range(len(Conditional))]
        
        # Sample from the conditional cdf of the first coordinate using inverse cdf method
        U2 = np.random.rand()
        j = len(Conditional) - 2
        while Conditional[j] > U2 and j > 0:
            j -= 1
        x = (U2 - Conditional[j])/(Conditional[j+1] - Conditional[j])*j/Nx + (Conditional[j+1] - U2)/(Conditional[j+1] - Conditional[j])*(j+1)/Nx
        
        # Save sampled point
        Points.append([x, y])
    
    return Points

def fit_gaussians(Points, n_Components):
    """From a set of points fit a gaussian mixture model and return a list of means and 3*covariance matrices (interpreted as 99% confindence region)"""
    
    # Fit gaussian mixture model to the given points
    gm = GaussianMixture(n_components=n_Components).fit(np.array(Points))
    
    # get means and covariance from the fit
    means = gm.means_.tolist()
    c = gm.covariances_
    
    # multiply covariances by 3 to get confidence region
    cov = [3*c[i,:,:] for i in range(c.shape[0])]
    
    return means, cov

def make_connection_matrix(Means, Covs, Map, alpha = 0.3, W = None, T = 1000):
    """Compute conectivity matrix for graph optimization"""
    
    # Initialize connectivity matrix of the needed shape
    N = len(Means)
    E = -1*np.ones((N, N))
    
    # If no W is given, use standard value
    if W is None:
        W = 0.01*np.array([[1,0],[0,1]])

    for i in range(N):
        for j in range(N):
            if i != j:
                # Test whether its reachable in straight line
                x1 = np.array(Means[i])
                x2 = np.array(Means[j])
                C1 = Covs[i]
                C2 = Covs[j]
                Path_Free = True
                # Collision check is done by checking grid points on the straight line
                for t in range(T):
                    P = x1 + t/(T-1)*(x2-x1)
                    C = C1 + t/(T-1)*(C2 - C1)
                    r = np.linalg.norm(C, ord=2)
                    
                    if bilinear_interpolation(P[0], P[1], Map) > 0:
                        Path_Free = False
                        break
                
                # If the straight line is collision free then write the distance of the nodes to the connectivity matrix
                if Path_Free:
                    dist = np.linalg.norm(x2 - x1)
                    Estimated_Cov = C1 + dist*W
                    dist_Covs = abs(np.log(np.linalg.det(Estimated_Cov)) - np.log(np.linalg.det(C2)))
                    Value = dist + alpha*dist_Covs
                    E[i,j] = Value
    return E

class Graph:
    """Graph class for the application of Dijkstra algorithm"""
    def __init__(self, Means, Covs, EdgeMatrix):
        # Initialize empty graph stored as dictionary
        self.graph = dict()

        N = min(len(Means), len(Covs))

        for i in range(N):
            # Store connectivity dictionaries for the indices of given nodes
            self.graph[str(i+1)] = dict()
            for j in range(N):
                # if the Edge matrix is -1 the nodes are not connected
                if i != j and EdgeMatrix[i,j] >= 0:
                    # if the nodes are connected store their edge value into the graph dictionary
                    self.graph[str(i+1)][str(j+1)] = EdgeMatrix[i,j]

        # The node with the largest index is considered the source for the Dijkstra algorithm
        self.source = str(N)

    def shortest_distances(self):
        # Initialize the values of all nodes with infinity
        distances = {node: float("inf") for node in self.graph}
        distances[self.source] = 0  # Set the source value to 0

        # Initialize a priority queue
        pq = [(0, self.source)]
        heapify(pq)

        # Create a set to hold visited nodes
        visited = set()

        while pq:  # While the priority queue isn't empty
            current_distance, current_node = heappop(pq)  # Get the node with the min distance
            if current_node in visited:
                continue  # Skip already visited nodes
            visited.add(current_node)  # Else, add the node to visited set

            for neighbor, weight in self.graph[current_node].items():
                # Calculate the distance from current_node to the neighbor
                tentative_distance = current_distance + weight
                if tentative_distance < distances[neighbor]:
                    distances[neighbor] = tentative_distance
                    heappush(pq, (tentative_distance, neighbor))
        return distances

def get_path_index(Distance_Dict, EdgeMatrix):
    """Given the computed value graph from Dijkstra and the connectivity matrix compute the index order of the shortest path according to the values in the graph starting from index '1'"""
    
    # Initialize path
    Path = list()
    
    # Starting node is '1'
    Path.append("1")
    
    # Store number of nodes
    N = len(Distance_Dict)
    
    # As long as the source node is not reached by the graph
    while Path[-1] != str(N):
        Candidates = list()
        for j in range(N):
            # Check if node is reachable from the current node
            if EdgeMatrix[int(Path[-1])-1, j] >= 0:
                # if reachable then store index to candidacy list with its value in the graph
                Candidates.append([str(j+1), Distance_Dict[str(j+1)]])
                
        # next node in the graph is the one with minimal distance to the source according to the graph
        Point = min(Candidates, key = lambda x: x[1])
        Path.append(Point[0])
    return Path

def construct_path_from_image(Img, n_sample_points, n_components, Start_Point, End_Point, Map, alpha, W):
    """Reconstructs optimal path from the Img produced by the neural network"""
    
    # Compute sample points from the density function given by the image
    # Note the computational time it takes for this step
    Start_Time = time.time()
    s1 = time.time()
    P = sample_by_image(Img, n_sample_points)
    Sample_Time = time.time() - s1
    
    # Fit gaussian mixture model to the sampled points
    # store the means and covariances
    # note the computational time for this step
    s2 = time.time()
    M, C = fit_gaussians(P, n_components)    
    Means = [Start_Point.tolist()] + M + [End_Point.tolist()]
    
    # Add covariances for the start and end point of the path as fixed values
    Covs = [0.001*np.array([[1,0],[0,1]])] + C + [0.001*np.array([[1,0],[0,1]])]
    GaussianFit_Time = time.time() - s2
        
    # Compute the connectivity matrix of the nodes given by the means and covariances
    # Computes the edge values as the distance function defined on the space of means and covariances
    # Note the computational time
    s3 = time.time()
    E = make_connection_matrix(Means, Covs, Map, alpha, W)
    ConstructGraph_Time = time.time() - s3

    # Construct the graph using the connectivity matrix and means and covariances as nodes
    # note the computational time
    s4 = time.time()
    G = Graph(Means, Covs, E)
    D = G.shortest_distances()
    GraphOpt_Time = time.time() - s4
    
    # If the distance to the source at the starting position is infinity, the source node is not reachable from the starting point
    # This happens of the connectivity of the graph is too sparse to produce a fesible path
    if np.isinf(D["1"]):
        Success = False
        Path_Length = None
        Points = list()
        
    else:
        Success = True
        
        # If there is a feasible path, compute the index list of the optimal feasible path
        I = get_path_index(D, E)
        IDX_List = [int(x)-1 for x in I]
        
        # From the index list get the mean of covariance matrices of the optimal path
        Points = [[Means[i], Covs[i]] for i in IDX_List]
        
        # Compute the path length of the reconstructed path
        Path_Length = 0
        for i in range(len(IDX_List)-1):
            Path_Length += np.linalg.norm(np.array(Means[i+1]) - np.array(Means[i]))
    
    # note the total computational time
    Computation_Time = time.time() - Start_Time
    return Success, Path_Length, Points, Computation_Time, Sample_Time, GaussianFit_Time, ConstructGraph_Time, GraphOpt_Time
    

def reconstruct_path(n_components, n_sample_points, alpha = 0.3, W = None):
    # W is the matrix that describes growth of covariance matrices
    # If no W is given use the standard value
    if W is None:
        W = 0.01*np.array([[1,0],[0,1]])
    
    # Create log file and folder to store results
    os.makedirs("0_Validation/Path_Reconstruction_"+str(n_components)+"/Images", exist_ok=True)
    with open("0_Validation/Path_Reconstruction_"+str(n_components)+"/Path_Reconstruction_Log.txt", "w") as f:
        f.write("IDX\tSuccess_Approx\tPath_Length_Approx\tComputationTime_Approx\tSuccess_Label\tPath_Length_Label\tComputationTime_Label\n")
    
    # List of angles used to plot covarinace matirces as ellipses 
    theta = np.linspace(0, 2*np.pi, 1000)
    
    # Create lists to store computed information
    Path_Length_Data = list()
    Success_Data = list()
    DetailedTime = list()
        
    # Iterate over test data set
    for IDX in tqdm(range(1, 101), desc = "Reconstructing Paths"):
        # load pre-stored data of input, label and network prediction
        Map = np.loadtxt("0_Validation/Data/Map_"+str(IDX)+".csv")
        Label = np.loadtxt("0_Validation/Data/Label_"+str(IDX)+".csv")
        Goal = 1*(np.loadtxt("0_Validation/Data/Goal_"+str(IDX)+".csv") < 0.03)
        Start = 1*(np.loadtxt("0_Validation/Data/Start_"+str(IDX)+".csv") < 0.0001)
        Img = np.loadtxt("0_Validation/Data/Prediction_"+str(IDX)+".csv")
        
        # Remove the target area from the label and network prediction
        # We use an endpoint in the target area and therefore do not sample too many additional points in this region 
        Img[Goal > 0] = 0
        Label[Goal > 0] = 0
        
        # Compute the starting point from the input data
        Start_Point = np.zeros(2)
        for i in range(Start.shape[0]):
            for j in range(Start.shape[1]):
                Start_Point += Start[i,j]*np.array([i/(Start.shape[0]-1),j/(Start.shape[1]-1)])
        Start_Point /= np.sum(Start)
        
        # Compute the center point of target region to use as an end-point of the path
        End_Point = np.zeros(2)
        for i in range(Goal.shape[0]):
            for j in range(Goal.shape[1]):
                End_Point += Goal[i,j]*np.array([i/(Goal.shape[0]-1),j/(Goal.shape[1]-1)])
        End_Point /= np.sum(Goal)
        
        # Reconstruct the optimal paths using label and estimated image
        Success_Approx, Path_Length_Approx, Points_Approx, Computation_Time_Approx, Sample_Time, GaussianFit_Time, ConstructGraph_Time, GraphOpt_Time = construct_path_from_image(Img, n_sample_points, n_components, Start_Point, End_Point, Map, alpha, W)
        Success_Label, Path_Length_Label, Points_Label, Computation_Time_Label, ST, GT, CGT, GOT = construct_path_from_image(Label, n_sample_points, n_components, Start_Point, End_Point, Map, alpha, W)
        
        # Mark down the measured cpu time for different steps in the algorithm
        DetailedTime.append([Sample_Time, GaussianFit_Time, ConstructGraph_Time, GraphOpt_Time])
        
        # Write results into log file
        with open("0_Validation/Path_Reconstruction_"+str(n_components)+"/Path_Reconstruction_Log.txt", "a") as f:
            f.write(str(IDX)+"\t"+str(Success_Approx)+"\t"+str(Path_Length_Approx)+"\t"+str(Computation_Time_Approx)+"\t"+str(Success_Label)+"\t"+str(Path_Length_Label)+"\t"+str(Computation_Time_Label)+"\n")
        
        # If path reconstruction is successful mark down computed path length
        if Success_Approx and Success_Label:
            Path_Length_Data.append([Path_Length_Approx, Path_Length_Label, Path_Length_Approx/Path_Length_Label])
        
        # Mark down whether reconstruction was successful
        Success_Data.append([1*Success_Approx, 1*Success_Label])
        
        # Split collected data for plotting
        Path_Approx_X = [p[0][0] for p in Points_Approx]
        Path_Approx_Y = [p[0][1] for p in Points_Approx]
        Path_Label_X = [p[0][0] for p in Points_Label]
        Path_Label_Y = [p[0][1] for p in Points_Label]
        
        # Plot collected results
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.contourf(Img, extent = [0,1,0,1])
        plt.contour(Map, extent = [0,1,0,1], colors = 'white')
        plt.scatter(Start_Point[1], Start_Point[0], color = "green")
        plt.scatter(End_Point[1], End_Point[0], color = "red")
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        
        plt.subplot(2, 2, 2)
        plt.contourf(Img, extent = [0,1,0,1])
        plt.contour(Map, extent = [0,1,0,1], colors = 'white')
        plt.scatter(Start_Point[1], Start_Point[0], color = "green")
        for i in range(len(Points_Approx)):
            plt.scatter(Points_Approx[i][0][1], Points_Approx[i][0][0])
            eigenvalues, eigenvectors = np.linalg.eig(Points_Approx[i][1])
            ellipsis = (np.sqrt(2*eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]
            plt.plot(Points_Approx[i][0][1]+ellipsis[1,:], Points_Approx[i][0][0]+ellipsis[0,:])
        plt.scatter(End_Point[1], End_Point[0], color = "red")
        plt.plot(Path_Approx_Y, Path_Approx_X, color = "red")
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        
        plt.subplot(2, 2, 3)
        plt.contourf(Label, extent = [0,1,0,1])
        plt.contour(Map, extent = [0,1,0,1], colors = 'white')
        plt.scatter(Start_Point[1], Start_Point[0], color = "green")
        plt.scatter(End_Point[1], End_Point[0], color = "red")
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        
        plt.subplot(2, 2, 4)
        plt.contourf(Label, extent = [0,1,0,1])
        plt.contour(Map, extent = [0,1,0,1], colors = 'white')
        plt.scatter(Start_Point[1], Start_Point[0], color = "green")
        for i in range(len(Points_Label)):
            plt.scatter(Points_Label[i][0][1], Points_Label[i][0][0])
            eigenvalues, eigenvectors = np.linalg.eig(Points_Label[i][1])
            ellipsis = (np.sqrt(2*eigenvalues[None,:]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]
            plt.plot(Points_Label[i][0][1]+ellipsis[1,:], Points_Label[i][0][0]+ellipsis[0,:])
        plt.scatter(End_Point[1], End_Point[0], color = "red")
        plt.plot(Path_Label_Y, Path_Label_X, color = "red")
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        
        plt.savefig("0_Validation/Path_Reconstruction_"+str(n_components)+"/Images/img_"+str(IDX)+".png")
    
    # Boxplot of absolute path length    
    Approx = [x[0] for x in Path_Length_Data]
    Label = [x[1] for x in Path_Length_Data]
    
    with open("StoredData/Data_Estimate.csv", "w") as f:
        f.write(",".join([str(x) for x in Approx]))
    
    with open("StoredData/Data_Label.csv", "w") as f:
        f.write(",".join([str(x) for x in Label]))
    
    plt.clf()
    plt.boxplot([Approx, Label])
    plt.xticks([1, 2], ['Estimate', 'Label'])
    plt.title("Path Length Comaprison")
    plt.savefig("0_Validation/Path_Reconstruction_"+str(n_components)+"/Absolute_Path_Length.png")
    
    # Boxplot of relative path length
    plt.clf()
    plt.boxplot([x[2] for x in Path_Length_Data])
    plt.title("Relative Path Length of Estimate")
    plt.savefig("0_Validation/Path_Reconstruction_"+str(n_components)+"/Relative_Path_Length.png")
    
    # Log success Rates
    Success_Rate_Approx = np.mean([x[0] for x in Success_Data])
    Success_Rate_Label = np.mean([x[1] for x in Success_Data])
    
    with open("0_Validation/Path_Reconstruction_"+str(n_components)+"/Success_Rates.txt", "w") as f:
        f.write("Aprox:\t"+str(100*Success_Rate_Approx)+"%\n")
        f.write("Label:\t"+str(100*Success_Rate_Label)+"%\n")
        
    with open("0_Validation/Path_Reconstruction_"+str(n_components)+"/PathLength.txt", "w") as f:
        f.write("Estimates are on average "+str(np.mean([x[2] for x in Path_Length_Data]))+" times the length of reconstructed label")
    
    return DetailedTime
    
def append_network_times(Folder):
    """Append measured network time to an existing log file as a new column"""
    
    # Define used weights for the network and the data folder used to measure foward call time
    WeightFile = "0_UNet_Results_2/Weights/95.pt"
    DataFolder = "FormattedData"
    
    # Initialize neural network
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    Net = UNet(3, 1, 3, 5, device)
    
    # Load weightfile into neural network
    Net.load(WeightFile)
    
    # Initialize Data loader
    D = DataLoader(32, DataFolder)
    
    # Load Test data from the data loader
    X_Test, Y_Test = D.get_test()
    N = X_Test.size(0)
        
    # Create list for noted times
    Times = list()
    
    # First entry of the list will be the column name of the added column in the log file
    Times.append("Network Time")
    
    # First application in the network is usually faster
    # To erase a bias in the measurement we first apply all inputs to the network without measure time
    for i in range(N):
        Est = Net.forward(X_Test[[i], :, :, :])
        
    # Measure forward call time of the neural network for all test data points separately
    for i in range(N):
        Start_Time = time.time()
        Est = Net.forward(X_Test[[i], :, :, :])
        Computation_Time = time.time() - Start_Time
        Times.append(Computation_Time)
    
    # Load the existing data from the log file
    with open(Folder+"/Path_Reconstruction_Log.txt", "r") as f:
        FileContent = "\n".join([x.strip("\n")+"\t"+str(Times[n]) for n,x in enumerate(f.readlines())])
    
    # Append the marked down times to the log file as a new column
    with open(Folder+"/Path_ReconstruM, C = fit_gaussians(P, n_components)ction_Log.txt", "w") as f:
        f.write(FileContent)
        
if __name__ == "__main__":
    
    #####################################################################################################
    # This section create images of the input data for illustration purposes                            #
    #####################################################################################################
    
    D = DataLoader(32, "FormattedData")
    X, Y = D.get_test()
    
    for i in tqdm(range(X.shape[0])):
        plt.clf()
        plt.subplot(2, 2, 1)
        plt.imshow(X[i, 0, :, :])
        plt.title("Obstacle Map")
        plt.subplot(2, 2, 2)
        plt.imshow(X[i, 1, :, :])
        plt.title("Initial Position")
        plt.subplot(2, 2, 3)
        plt.imshow(X[i, 2, :, :])
        plt.title("Target Region")
        plt.subplot(2, 2, 4)
        plt.imshow(Y[i, :, :])
        plt.title("Computed Solution")
        plt.savefig("0_Validation/InputPlots/"+str(i+1)+".png")
    
    #####################################################################################################
    # This section performs path reconstruction for different number of gaussian components for testing #
    #####################################################################################################
    
    n_component_list = [5, 10, 20, 50, 100]
    
    for n in n_component_list:
        reconstruct_path(n, 1000)
        append_network_times("0_Validation/Path_Reconstruction_"+str(n))
        
    #####################################################################################################
    # This section gives an overview of the cpu time of different part of the algorithm                 #
    #####################################################################################################
    
    Times = reconstruct_path(10, 1000)
    
    SampleMean = np.mean([x[0] for x in Times])
    GaussianFit = np.mean([x[1] for x in Times])
    ConstGraph = np.mean([x[2] for x in Times])
    GraphOpt = np.mean([x[3] for x in Times])
    S = sum([SampleMean, GaussianFit, ConstGraph, GraphOpt])
    
    print("Sample", "GaussianFit", "ConstGraph", "GraphOpt")
    print(SampleMean, GaussianFit, ConstGraph, GraphOpt)
    print(SampleMean/S, GaussianFit/S, ConstGraph/S, GraphOpt/S)
    
    #####################################################################################################
    # Makes a boxplot of total cpu time of the algorithm                                                #
    #####################################################################################################
        
    with open("0_Validation/Path_Reconstruction_10/Path_Reconstruction_Log.txt", "r") as f:
        L = [x.strip("\n").split("\t") for x in f.readlines()]
    
    TimeData = [float(L[i][3])+float(L[i][-1]) for i in range(1, len(L))]
    
    with open("StoredData/Time_Data.csv", "w") as f:
        f.write(",".join([str(x) for x in TimeData]))
    
    
    plt.clf()
    plt.boxplot(TimeData)
    plt.title("Total computation time in [s]")
    plt.xticks([1], ["Network Prediction and Path Reconstruction"])
    
    #####################################################################################################
    # This section performs path reconstruction on the test data set and plots the results              #
    #####################################################################################################
    
    n_components = 10
    n_sample_points = 100
    
    os.makedirs("0_Validation/Paths/Images", exist_ok=True)
    
    with open("0_Validation/Path_Reconstruction_Log.txt", "w") as f:
        f.write("")
    
    for IDX in tqdm(range(1, 101), desc = "Reconstructing Paths"):
        Map = np.loadtxt("0_Validation/Data/Map_"+str(IDX)+".csv")
        Label = np.loadtxt("0_Validation/Data/Label_"+str(IDX)+".csv")
        Goal = 1*(np.loadtxt("0_Validation/Data/Goal_"+str(IDX)+".csv") < 0.03)
        Start = 1*(np.loadtxt("0_Validation/Data/Start_"+str(IDX)+".csv") < 0.0001)
        Img = np.loadtxt("0_Validation/Data/Prediction_"+str(IDX)+".csv")
        Img[Goal > 0] = 0
        Label[Goal > 0] = 0
    
        Start_Point = np.zeros(2)
        for i in range(Start.shape[0]):
            for j in range(Start.shape[1]):
                Start_Point += Start[i,j]*np.array([i/(Start.shape[0]-1),j/(Start.shape[1]-1)])
        Start_Point /= np.sum(Start)
        
        End_Point = np.zeros(2)
        for i in range(Goal.shape[0]):
            for j in range(Goal.shape[1]):
                End_Point += Goal[i,j]*np.array([i/(Goal.shape[0]-1),j/(Goal.shape[1]-1)])
        End_Point /= np.sum(Goal)
        
        P = sample_by_image(Img, n_sample_points)
        
        M, C = fit_gaussians(P, n_components)
        M = [[max(min(p[0], 1), 0), max(min(p[1], 1), 0)] for p in M]
        Full_Points = [[Start_Point[0], Start_Point[1]]] + M + [[End_Point[0], End_Point[1]]]
        Con = construct_graph(Full_Points, Img, Map)
        
        IDX_List = path_reconstruction(Con, Full_Points)
        
        if IDX_List:
            if len(IDX_List) < len(Full_Points):
                with open("0_Validation/Path_Reconstruction_Log.txt", "a") as f:
                    f.write("Estimate "+str(IDX)+"\t"+str(len(Full_Points) - len(IDX_List))+" Artifacts detected\n")
        else:
            with open("0_Validation/Path_Reconstruction_Log.txt", "a") as f:
                f.write("Estimate "+str(IDX)+"\tUnable to reconstruct Estimated Path\n")
        
        Path_Estimate = np.array([Full_Points[idx] for idx in IDX_List])
        
        P_Label = sample_by_image(Label, n_sample_points)
        
        M_Label, C_Label = fit_gaussians(P_Label, n_components)
        M_Label = [[max(min(p[0], 1), 0), max(min(p[1], 1), 0)] for p in M_Label]
        Full_Points_Label = [[Start_Point[0], Start_Point[1]]] + M_Label + [[End_Point[0], End_Point[1]]]
        Con_Label = construct_graph(Full_Points_Label, Label, Map)
        
        IDX_List_Label = path_reconstruction(Con_Label, Full_Points_Label)
        
        if IDX_List_Label:
            if len(IDX_List_Label) < len(Full_Points_Label):
                with open("0_Validation/Path_Reconstruction_Log.txt", "a") as f:
                    f.write("Label "+str(IDX)+"\t"+str(len(Full_Points_Label) - len(IDX_List_Label))+" Artifacts detected\n")
        else:
            with open("0_Validation/Path_Reconstruction_Log.txt", "a") as f:
                f.write("Label "+str(IDX)+"\tUnable to reconstruct Estimated Path\n")
        
        Path_Label = np.array([Full_Points_Label[idx] for idx in IDX_List_Label])
        
        np.savetxt("0_Validation/Paths/Estimate_"+str(IDX)+".csv", Path_Estimate)
        np.savetxt("0_Validation/Paths/Label_"+str(IDX)+".csv", Path_Label)