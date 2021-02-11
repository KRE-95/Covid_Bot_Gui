import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_U import bag_of_words, tokenize, stem
from model import Brain

# training data

with open('intents.json', 'r') as f:  # file
    intents = json.load(f)

all_words = []
tags = []
xy = []  # will hold the pattern and the tags.
# loop on our intent - to check if it exist in the list by using In the jason_file:
for intent in intents['intents']:  # sense in the jason file we have intents as key and then we have one array
    tag = intent['tag']  # with the all different tags , patterns and responses.
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:  # make an loop for pattern (for each single intent)
        # tokenize each word in the sentence
        w = tokenize(pattern)  # after import nktl_U, tokenize, stemer.
        # add to our words list
        all_words.extend(w)  # why we use extend and not append, because we don't want to use array inside a array.
        xy.append((w, tag))  # use of tuple, so it would it contain the pattern and the corresponding tag.

# define some ignore words:
ignore_words = ['?', '.', '!', ',', ':', ';', '!?', '#', '&', '^', '%']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# This help to remove any duplicated words,keep it more unique of selection of words - sorted.
all_words = sorted(set(all_words))
tags = sorted(set(tags))

"""
To check if it works:
=========================================================
print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)
=========================================================
"""

# create  X_list and Y data aka. train it.
X_train = []
y_train = []
# Loop over xy [], giving variable (not tuple, sense we have already given the func)
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)  # short name
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss require only class labels.
    label = tags.index(tag)
    y_train.append(label)
    """
    this helps both human and the bot to understand what information,
    the user input would be asked for, by labeling it, e.g  0  and  1.. nr for out label.
    """
# convert it to numpy  arrays  as np.
X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
num_epochs = 1000
batch_size = 8
learn_rate = 0.001
input_size = len(X_train[0])  # this has the same length as bag_of_words
hidden_size = 8
output_size = len(tags)  # number of diff class, tags
print(input_size, output_size)
"""
What I mean with same size , try this print statement, where you get 54 54 same length  and our jason file.
# print(input_size, len(all_words))
# print(output_size, tags)
"""


# implement the bag_of_words , import pytorch = torch:
# new dataset by making a class

class ChattyDataset(Dataset):  # data loading

    def __init__(self):
        self.n_samples = len(X_train)  # sort the number of sample
        self.x_data = X_train
        self.y_data = y_train

    # dataset[index]
    def __getitem__(self, index):  # This would allow indexing later.
        return self.x_data[index], self.y_data[index]

    # This would allow the length of our dataset _ we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# https://pytorch.org/docs/stable/data.html
dataset = ChattyDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
"""
I came across an error , where my software on mac couldn't comprehend num_workers = 2, first i had the speed on
two , but .... it didn't work because i wanted it to load faster. 
What is nice by putting as 0 , it becomes kind of universal code, where both  mac and windows 
can run it without changing it.
"""

# if this is windows the use cuda device, else cpu.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model.py
model = Brain(input_size, hidden_size, output_size).to(device)

"""
   This would help to automatically iterate for us, by making a data loader and get batch training.
   batch_size = number of training samples in one forward & backwards.
   epoch means 1 forward and backward pass of all training sample (would be used in the other script)
   e.g.  sample = 100 and batch_size 50 than the iteration would  and 2 epoch.

"""

"""
TRAINING_Pipeline - step 2
Then as a second step we design or we come up with the so we construct the loss an the optimize.
"""

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

"""
TRAINING_Pipeline - step 3
Last step we do our training loop, we start by doing our forward pass 
so here we compute or where we write a given  prediction e.g 0,70 (on Bot_Gui.py)
then we do the backward pass backward pass so we get the gradient.
"""
# train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # (This is an training loop)
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        """
        For every 100 step, we want to print current epoch which is epoch + 1 
        and all epc and loos = 4f for decimal ( This is an training loop)al values.
        """

print(f'final loss: {loss.item():.4f}')

# save the data, dict
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"  # pytorch, generate a file called data.pth
torch.save(data, FILE)  # save the data and file.

print(f'training complete. file saved to {FILE}')  # prints on the console (backend)

# for more info>\:  https://pytorch.org/docs/stable/data.html
# https://stackoverflow.com/questions/62950311/accuracy-per-epoch-in-pytorch


# I know the loss is suppose to add global , but I don't know how apply.
