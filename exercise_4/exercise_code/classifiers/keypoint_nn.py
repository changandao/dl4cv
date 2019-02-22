import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        
        
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        
        self.pool = nn.MaxPool2d(2)
        
        self.dropout1 = nn.Dropout2d(0.1)
        self.dropout2 = nn.Dropout2d(0.2)
        self.dropout3 = nn.Dropout2d(0.3)
        self.dropout4 = nn.Dropout2d(0.4)
        self.dropout5 = nn.Dropout2d(0.5)
        self.dropout6 = nn.Dropout2d(0.6)
        
        self.dense1 = nn.Linear(6400, 1000)
        self.dense2 = nn.Linear(1000, 1000)
        self.dense3 = nn.Linear(1000, 30)
        
#         self.conv = nn.Sequential(
#             conv1,
#             nn.ELU(),
#             pool,
#             dropout1,
            
#             conv2,
#             nn.ELU(),
#             pool,
#             dropout2,
            
#             conv3,
#             nn.ELU(),
#             pool,
#             dropout3,
            
#             conv4,
#             nn.ELU(),
#             pool,
#             dropout4
#         )
        
        self.fc = nn.Sequential(
            self.dense1,
            nn.ELU(),
            self.dropout5,
            
            self.dense2,
            nn.ReLU(),
            self.dropout6,
            
            self.dense3
        )
        
         # Custom weights initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         # Conv layers have weights initialized with random # drawn from uniform distribution
        #         m.weight = nn.init.uniform(m.weight, a=0, b=1) # nn.init.uniform_ does not seem to work for in-place change
        #     elif isinstance(m, nn.Linear):
        #         # FC layers have weights initialized with Glorot uniform initialization
        #         m.weight = nn.init.xavier_uniform(m.weight, gain=1)
        

        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################        
        x = F.elu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
            
        x = F.elu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)
            
        x = F.elu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout3(x)
        
        x = F.elu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout4(x)
        
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x
    

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
