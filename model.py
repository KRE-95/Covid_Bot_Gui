import torch.nn as nn

"""
TRAINING_Pipeline - step 1
About: The general training pipeline in pytorch so typically we have three steps so the 
First step is to design our model so we design, the number of inputs and outputs so input size and output size 
and then also we design the forward pass with all the different operations or all the different layers 

BUT FIRST!! import the neural network  as nn other you cannot recall these func.
the other steps on the script train.py 
"""


class Brain(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Brain, self).__init__()
        """
        THIS is important have, because when you recall in train.py. ill not work
        if you don't add it. than I mean super.
        """
        self.l1 = nn.Linear(input_size, hidden_size)  # make linear layers
        self.l2 = nn.Linear(hidden_size, hidden_size)  # watch the patterns
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()  # self activation function in between

    # Implement the forward pass func:
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # why we don't put activation her out = self.relu(out), were going let bot apply for us same goes softmax.
        return out

    # relu is activation

# https://discuss.pytorch.org/t/bidirectional-gru-lstm-error/21327/2
