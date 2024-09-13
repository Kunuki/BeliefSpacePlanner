import numpy as np
import os
from tqdm import tqdm
import random
import torch as t
import skfmm
import matplotlib.pyplot as plt
import random
from PIL import Image

def sign_distance(Array):
    """Takes an indicator array (Only values 0 and 1) and returns signed distance function (+/- euclidean distance to closes pixel with value 1/0)"""
    # use the distance function from skfmm to efficiently compute distances
    # the difference of the two will be the signied distance function
    Dist = skfmm.distance(Array) - skfmm.distance(1 - Array)
    
    # Normalize the resulting array
    # such that the min is 0 and the max is 1
    # this makes it easier for the network to learn
    m = np.min(Dist)
    M = np.max(Dist)
    Dist = (Dist - m)/(M-m)
    return Dist

def set_size(Array, size):
    """Format given Array to a specified size using image processing"""
    # load array as an image
    im = Image.fromarray(np.uint8(Array*255))
    
    # use image processing to resize the image
    im = im.resize(size)
    
    # cast the resized image back to an array
    Array = np.array(im)/255
    return Array

def get_index_list(Folder):
    """Returns list of indices of available data points in the given data folder."""
    
    # create lists to store indices of available data for every input and output array separately
    GoalIDX = list()
    ObsIDX = list()
    StartIDX = list()
    PathIDX = list()
    
    # iterate through the available files and check available indices
    # append available indices to the created lists
    for file in os.listdir(Folder + "/input_data/goal/"):
        file = file.split(".")[0].replace("goal", "")
        GoalIDX.append(int(file))
    for file in os.listdir(Folder + "/input_data/obstacle/"):
        file = file.split(".")[0].replace("obstacle", "")
        ObsIDX.append(int(file))
    for file in os.listdir(Folder + "/input_data/start/"):
        file = file.split(".")[0].replace("start", "")
        StartIDX.append(int(file))
    for file in os.listdir(Folder + "/output_data/"):
        file = file.split(".")[0].replace("output", "")
        PathIDX.append(int(file))
    
    # Cast list to sets to eliminate double entries
    # Sets are also used to compute intersection (i.e. indices where all data is available)
    GoalIDX = set(GoalIDX)
    ObsIDX = set(ObsIDX)
    StartIDX = set(StartIDX)
    PathIDX = set(PathIDX)
    
    # sort available indices
    Common = sorted(list(GoalIDX & ObsIDX & StartIDX & PathIDX))
    return Common
        
def load_data_idx(Folder, idx):
    """Load the data point with the specifed index in the given data folder"""
    
    # load the arrays of all input components and label separately
    goal = np.loadtxt(Folder + "/input_data/goal/goal"+str(idx)+".csv", delimiter=",")
    obstacle = np.loadtxt(Folder + "/input_data/obstacle/obstacle"+str(idx)+".csv", delimiter=",")
    start = np.loadtxt(Folder + "/input_data/start/start"+str(idx)+".csv", delimiter=",")
    path = np.loadtxt(Folder + "/output_data/output"+str(idx)+".csv", delimiter=",")
    
    # Check feasibility of data
    if np.max(start) > np.min(start):
        # compute start array as sign distance array
        start = sign_distance(1-start)
    else:
        return False, 0, 0
    
    # Check feasibility of data
    if np.max(goal) > np.min(goal):
        # compute start array as sign distance array
        goal = sign_distance(1-goal)
    else:
        return False, 0, 0
    
    # resize all components to make sure they all have the same size
    goal = set_size(goal, (225, 150))
    obstacle = set_size(obstacle, (225, 150))
    start = set_size(start, (225, 150))
    path = set_size(path, (225, 150))
    
    # Check feasibility of all data points (to explude empty data points)
    if np.max(start) > np.min(start) and np.max(goal) > np.min(goal) and np.max(obstacle) > np.min(obstacle) and np.max(path) > np.min(path):
        # Concatenate the input components into the full input array
        In = np.concatenate([obstacle.reshape((1, 1, obstacle.shape[0], obstacle.shape[1])), start.reshape((1, 1, start.shape[0], start.shape[1])), goal.reshape((1, 1, goal.shape[0], goal.shape[1]))], axis = 1)
        return True, In, path.reshape((1, path.shape[0], path.shape[1]))
    else:
        return False, 0, 0
    
def get_index_formatted(Folder):
    """Return list of indices of the available formatted data in the specified folder"""
    # Create list to store availabe indices of formatted data points
    IDX = list()
    
    # iterate over all files in the given folder
    for file in os.listdir(Folder):
        # extract index number from file name
        idx = int(file.split(".")[0].split("_")[1])
        if idx not in IDX:
            # append index to the index list
            IDX.append(idx)
    # sort the collected indices
    IDX = sorted(IDX)
    return IDX
        
class DataLoader:
    """Data Loader handling data loading, batching, and train test split of a given data folder"""
    def __init__(self, BatchSize, Folder):
        
        # store batch size and data folder from input
        self.BatchSize = BatchSize
        self.Folder = Folder
        
        # get available indices of the provided data folder
        IDX = get_index_formatted(Folder)
        
        # reserve the last 100 indices as test data
        self.IDX = IDX[:-100]
        
        # the reminder of the indices is used as training data
        self.TestIDX = IDX[-100:]
        
        # compute the total number of batches given the number of available data points and desired batch size
        self.N = int(len(self.IDX)/BatchSize)
        if self.N*BatchSize < len(self.IDX):
            self.N += 1
        
    def get_test(self):
        """Return the test data of the data folder"""
        
        # create list to store loaded data
        Inp = list()
        Out = list()
        
        # iterate over list of indices of the test data
        for idx in self.TestIDX:
            # load input and output array
            I = np.load(self.Folder+"/In_"+str(idx)+".npy").reshape((1, 3, 150, 225))
            O = np.load(self.Folder+"/Out_"+str(idx)+".npy").reshape((1, 150, 225))
            
            # append them to the list
            Inp.append(I)
            Out.append(O)
            
        # cast data to pytorch tensor and return 
        return t.tensor(np.concatenate(Inp, axis = 0)), t.tensor(np.concatenate(Out, axis = 0))
        
    def __iter__(self):
        """Iterate over training data batches"""
        
        # when iteration over training data is initialize the index list of the training data is shuffled for stochastic gradient descent
        random.shuffle(self.IDX)
        
        # initialize the batch number of the iteration
        self.n = 0
        return self
        
    def __next__(self):
        # if the current batch number is smaller than the computed maximal batch number
        if self.n < self.N:
            # get the index assiciated to the next batch
            Batch = self.IDX[self.n*self.BatchSize:(self.n+1)*self.BatchSize]
            
            # create lists to store loaded data
            Inp = list()
            Out = list()
            
            
            for idx in Batch:
                # load data for the current index batch
                I = np.load(self.Folder+"/In_"+str(idx)+".npy").reshape((1, 3, 150, 225))
                O = np.load(self.Folder+"/Out_"+str(idx)+".npy").reshape((1, 150, 225))
                
                # append the loaded data to the lists
                Inp.append(I)
                Out.append(O)
            
            # update current batch number
            self.n += 1
            # cast data to pytorch tensor and return 
            return t.tensor(np.concatenate(Inp, axis = 0)), t.tensor(np.concatenate(Out, axis = 0))
        else:
            raise StopIteration
        
    def __len__(self):
        # length of the iteration is the number of batches
        # this is mainly used for progress bars
        return self.N