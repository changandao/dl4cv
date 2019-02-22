import torch
import torch.nn as nn
#import torch.function as F

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
#         w_ih = nn.Parameter(torch.Tensor(input_size, hidden_size))
#         w_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         b_ih = nn.Parameter(torch.Tensor(hidden_size))
#         b_hh = nn.Parameter(torch.Tensor(hidden_size))
#         layer_params = (w_ih, w_hh, b_ih, b_hh)

#         suffix = '_reverse' 
#         param_names = ['weight_ih_l{}{}', 'weight_hh_l{}{}']
#         param_names += ['bias_ih_l{}{}', 'bias_hh_l{}{}']
#         param_names = [x.format(1, suffix) for x in param_names]
#         self._all_weights.append(param_names)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        
        h_seq = x.new_zeros(x.size(0),
                             x.size(1), self.hidden_size)
        # h_seq = torch.zeros((x.size()[0],x.size()[1],self.hidden_size))
        seq_len = x.size(0)
        
        h_tempt = torch.zeros(1,
                             x.size(1), self.hidden_size)
        for i in range(seq_len):
            h_tempt = self.activation(self.fc2(h_tempt) + self.fc1(x[i]))
            h_seq[i] = h_tempt
        h = h_tempt
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
            
    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq=[]
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        pass

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , (h, c)
    

class RNN_Classifier(torch.nn.Module):
    def __init__(self, classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()
        ############################################################################
        #  TODO: Build a RNN classifier                                            #
        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size,hidden_size)
        self.fc = nn.Linear(hidden_size, classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def forward(self, x):
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        out, _ = self.rnn(x)
        out = self.fc(out[-1])
        return out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class LSTM_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128):
        super(LSTM_Classifier, self).__init__()
        ############################################################################
        #  TODO: Build a LSTM classifier                                           #
        ############################################################################
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size,hidden_size)
        self.fc = nn.Linear(hidden_size, classes)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def forward(self, x):
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        out, _ = self.rnn(x)
        out = self.fc(out[-1])
        return out

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
        