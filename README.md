# BeliefSpacePlanner

This repository contains a project focused on neural network training and path reconstruction. The core functionality is based on a custom UNet architecture that takes start, goal, and obstacle maps as input and generates a predicted path. The repository also includes methods for evaluating the trained network on new data and reconstructing paths using Gaussian Mixture Models (GMM) and graph optimization.

## Features

### Neural Network Training
* **UNet Architecture:** A UNet-based neural network is trained to predict optimal paths given start, goal, and obstacle inputs.
* **Data Loading:** Training and test data are loaded through a custom data loader that formats input data and applies a signed distance transform for the start and goal positions.
* **Training:** The model is trained using binary cross-entropy loss, with training and test loss logged and visualized over epochs.
* **Weights Saving/Loading:** The network supports saving and loading weights during training, allowing for continued training from checkpoints.

### Path Reconstruction
* **Image-Based Path Prediction:** The network outputs predicted paths, which are used as a density map to sample points.
* **Gaussian Mixture Model (GMM):** Sampled points are used to fit a GMM, which clusters path points based on the predicted density. This allows for a probabilistic representation of the path in terms of means and covariances.
* **Graph Construction:** Using the GMM means as nodes, a graph is built where edges represent possible paths between nodes. A distance function is calculated for each pair of nodes based on Euclidean distance and covariance information.
* **Graph Optimization:** The shortest path between the start and goal points is found using Dijkstra’s algorithm on the constructed graph.
Path Validation: The reconstructed path is evaluated against the ground truth, comparing path length and accuracy.

## Usage

### Training the Model
1. The code contains capabilities to prepare the training and test data in the expected format
2. Formatted training data can be found at .
3. Run the training script to start training the UNet model.
   ```bash
   python network.py
   ```
### Path Reconstruction
Once the model is trained, you can perform path reconstruction by running the reconstruct_path function, which reconstructs the optimal path from the model's predictions.
```python
reconstruct_path(n_components=10, n_sample_points=1000)
```
This will generate paths based on the trained model's predictions and compare them to the ground truth.

### Test Results and Evaluation

The repository also includes utilities for:
* Loading test data and evaluating the model's performance.
* Visualizing predicted paths versus the ground truth paths.
* Saving evaluation metrics such as path length and reconstruction success.

## Requirements
* PyTorch
* NumPy
* scikit-learn
* tqdm
* skfmm
* PIL
* matplotlib
