# # # # # # # # # # # # # # # # # # # # #
#
# First attempt at ML model
# 

#Load packages. Using PyTorch by choice
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


# Set parameters
#
# - Image dimensions (pixel x pixel dimensions). Can also edit channels of input, but usually RGB.
#       Final input shape (height, width, 3)
#
# - Batch size. Number of training samples (images) that will be passed through the NN at once before
#       the models internal parameters are updated
#
# - Data Directories
#
img_height = 224
img_width = 224
batch_size = 24
train_dir = 'PATH'
validation_dir = 'PATH'



# Data Preprocessing
#
# We use the 'transforms' method of PyTorch to prepare our data for processing
#
# Random Horizontal Flip reverses each image along vertical axis. This has a couple of upsides:
#   - Flipping images effectively doubles training examples w/out new data creation
#   - Since GW Lensing is invariant for left/right orientation, flipping helps model generalization
#   - General Overfitting prevention (see above point)
#
# Random Rotation is also used for similar reasons.
# 
transform_train = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(), 
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

transform_validation = transforms.Compose([
    transforms.Resize((img_height, img_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])



# Load Data & preprocess
# 
# Num_workers refers to the amount of seperate processes used to load data. Ie for 2, the data gets split
#   - Increased num_workers can sometimes how downsides, based on various factors such as complexity of 
#       transformations, hardware being used, etc
#   - Of course, less processes also means data loading becomes bottlenecked
# 
# -> An optimal value for num_workers is necessary. A good start is the number of CPU cores
#       import psutil
#       num_workers = psutil.cpu_count(logical = True/False)
#   Where True = # of logical CPU cores and False = physical CPU cores
#
train_dataset = torchvision.datasets.ImageFolder(root = train_dir, transform = transform_train)
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

validation_dataset = torchvision.datasets.ImageFolder(root = validation_dir, transform = transform_validation)
validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = False, num_workers = 2)



# Create the CNN structure
# 
# CNN class inherits from nn.Module, the base class for all NN in pytorch
#
# __init__ Initializes all netowrk layers and params
#
# Forward method defines how data flows through the network layers
# 
# -> For image classification and object detection, convolutional layers are more critical than dense layers.
#       - These layers capture and learn spatial heirarchies and extract relecant features
#       - More convolutional layers, but not too much since overfitting is a probelm
# 
class CNN(nn.Module): 
    def __init__(self): # Initializes netowrk layers and params
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3,32,3, padding = 1)# First convolutional layer
        # 3 = 3 input channels (RGB). 32 = Number of output channels (filters). 3 = Size of Conv. Kernel (3x3)
        # padding 1 = adds 1 pixel border of 0s around input -> keeps output same size as input

        self.pool = nn.MaxPool2d(2,2) # Maximum Pooling layer
        # 2, 2 = size of Window, Stride. Both set to 2x2 -> reduces each dimension by half

        self.conv2 = nn.Conv2d(32, 64, 3, padding = 1) # Second Conv. Layer
        # input needs to be same size as last output ie 32

        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1) # Third Conv. Layer

        self.fc1 = nn.Linear(128 * 28 * 28, 512) # First fully connected AKA dense layer
        # 128*28*28 = Number of input features, flattened output from conv. layers. 
        # 512 = Number of output features (neurons)

        self.fc2 = nn.Linear(512, 2) # Second fully connected Layer
        # 512 = input features from previous layer. 2 = output features, for binary classification

        self.dropout = nn.Dropout(0.5) # Defines dropout layer
        # 0.5 = 50% of neurons will randomly be set to 0 during training -> prevents overfitting

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Applies first conv. layer, then applies ReLU activation function
        # ReLU introduces non-linearity into model -> allows it to learn complex patterns

        x = self.pool(F.relu(self.conv2(x))) # Applies second conv. layer
        x = self.pool(F.relu(self.conv3(x))) # ^

        x = x.view(-1,128*28*28) # Flattens output tensor from Conv. Layers to 1D tensor. This is necessary 
        # in order to feed it to the dense layers

        x = F.relu(self.fc1(x)) # Applies first dense layer, then ReLU activation
        x = self.dropout(x) # Applies dropout to output of dense layer
        x = self.fc2(x) # Applies second dense layer
        
        return x

model = CNN()



# Defiine Loss Function, Optimizer
#
# CrossEntropyLoss uses multiple Log loss functions and combines them into a single class
#   - Input: Raw scores from models output layer
#   - Target: True class labels
#
# Optim.Adam -> Adaptive Moment Estimation
#   - Model parameters: Method that retuns that returns all params (weights & biases) that need to be 
#       optimized. Also ensures all learnable params are included in optimization process
#   - lr: Learning rate. Hyperparameter, which means we choose. 
#       Small learning rate -> more precise adjustments, but longer time for convergence
#
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001) # Hyperparameter learning rate



# Train Model
#
num_epochs = 25 # Defines how many times Training data is passed through NN

for epoch in range(num_epochs):
    running_loss = 0.0 # Keeps track of loss for running epoch

    for inputs, labels in train_loader: # Iterates over training data in batches
        # Train_Loader provides batches of data from training data set 
        # Inputs = a batch of input images
        # Labels = ground truth labels for input images

        optimizer.zero_grad() # clears gradients from last loop

        outputs = model(inputs) # Computes model's predictions by performing forward pass through NN

        loss = criterion(outputs, labels) # Computes Loss based on prediction
        loss.backward() # Computes gradients of loss WRT the model parameters. "Backward pass"

        optimizer.step() # Updates models parameters using computed gradients, adjusts weights to minimize loss

        running_loss += loss.item() # Updates running loss.
        # Converts loss tensor to python number, easier for accumulation
    
    print(f'[Epoch {epoch + 1}, Batch {i+1}] loss: {running_loss/len(train_loader):.3f}')
    # prints epoch loss

print('Finished Training') 



# Evaluate Model
# 
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in validation_loader:
        # images, labels = data

        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

print(f'Accuracy of the NN based on validation set: {100 * correct / total:.2f}%')
