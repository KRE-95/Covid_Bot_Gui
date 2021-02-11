import random
import json
import torch
from model import Brain
from nltk_U import bag_of_words, tokenize
from tkinter import *
from random import choice
import os

# for windows, doesnt work for mac
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data: intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = Brain(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "CovidBot:"
# print("Let's talk business ! (type 'quit' to exit)")


# while True:
#    # sentence = from jason file
#    sentence = input("You: ")
#    if sentence == "quit":
#        break
def getreply(x):
    sentence = str(x)
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    # possibility
    """
    On the model.py we didn't add softmax or activation the reason, we want the CovidBot to do for us.
    """
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.70:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                # print(f"{bot_name}: {random.choice(intent['responses'])}")
                return f"{bot_name}: {random.choice(intent['responses'])}"
    else:
        # print(f"{bot_name}: I'm sorry , I didn't get what you wrote.")
        return f"{bot_name}: I'm sorry , I didn't get what you wrote."


# create tkinter
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    bot_reply = (getreply(msg))  # bot reply
    if msg != '':
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.insert(END, bot_reply + '\n\n')
        ChatBox.config(foreground="#446665", font=("Verdana", 12))
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)


class NewWindow(Toplevel):  # import OS
    def __init__(self, master=None):
        super().__init__(master=master)
        self.title("Covid_bot")
        self.geometry("500x500")
        self.resizable(width=FALSE, height=FALSE)

        self.menubar = Menu(self)
        self.filemenu = Menu(self.menubar, tearoff=0)
        self.filemenu.add_command(label="New", command=startnew)
        self.filemenu.add_command(label="Save", command=self.savechat)
        self.filemenu.add_separator()
        self.filemenu.add_command(label="Exit", command=self.quit)
        self.menubar.add_cascade(label="Menu", menu=self.filemenu)

        # Create Chat window
        self.ChatBox = Text(self, bd=0, bg="white", height="8", width="50", font="Arial", )
        self.ChatBox.config(state=DISABLED)
        # Bind scrollbar to Chat window
        self.scrollbar = Scrollbar(self, command=self.ChatBox.yview, cursor="heart")
        self.ChatBox['yscrollcommand'] = self.scrollbar.set
        # Create Button to send message
        self.SendButton = Button(self, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                                 bd=0, bg="#f9a602", activebackground="#3c9d9b", fg='#000000', command=self.send)
        # Create the box to enter message
        self.EntryBox = Text(self, bd=0, bg="white", width="29", height="5", font="Arial")
        # EntryBox.bind("<Return>", send)
        # Place all components on the screen

        self.scrollbar.place(x=376, y=6, height=386)
        self.ChatBox.place(x=6, y=6, height=386, width=370)
        self.EntryBox.place(x=128, y=401, height=90, width=265)
        self.SendButton.place(x=6, y=401, height=90)
        self.config(menu=self.menubar)

    def send(self):
        msg = self.EntryBox.get("1.0", 'end-1c').strip()
        # print(self.ChatBox.get("1.0", 'end-1c').strip())
        self.EntryBox.delete("0.0", END)
        bot_reply = (getreply(msg))
        if msg != '':
            self.ChatBox.config(state=NORMAL)
            self.ChatBox.insert(END, "You: " + msg + '\n\n')
            self.ChatBox.insert(END, bot_reply + '\n\n')
            self.ChatBox.config(foreground="#446665", font=("Verdana", 12))
            self.ChatBox.config(state=DISABLED)
            self.ChatBox.yview(END)

    def savechat(self):
        getall_chat = str(self.ChatBox.get("1.0", 'end-1c').strip())
        while True:
            catid = "ABCDEFGHIJKLMNOPQUSTUVWXYZ1234567890"  # use random numbers to save a file.
            filenamee = choice(catid) + choice(catid) + choice(catid) + choice(catid) + choice(catid) + choice(
                catid) + choice(catid) + ".txt"
            if os.path.isfile(filenamee) != True:
                # catid = "0a1b2c3d4e5f6g7h8i8j9klmnopqrstuvwxyz1234567890" filenamee = choice(catid)+choice(
                # catid)+choice(catid)+choice(catid)+choice(catid)+choice(catid)+choice(catid)+".txt"
                open(filenamee, "w").write(getall_chat)
                os.startfile(filenamee)
                break
            else:
                pass


def startnew(): NewWindow(root)


def savechat():
    getall_chat = str(ChatBox.get("1.0", 'end-1c').strip())
    while True:
        catid = "ABCDEFGHIJKLMNOPQUSTUVWXYZ1234567890" # Makes an prefer name file by those words
        filenamee = choice(catid) + choice(catid) + choice(catid) + choice(catid) + choice(catid) + choice(
            catid) + choice(catid) + ".txt"
        if os.path.isfile(filenamee) != True:
            open(filenamee, "w").write(getall_chat)
            os.startfile(filenamee)
            break
        else:
            pass


root = Tk()
root.title("Covid_bot")
root.geometry("500x500")
root.resizable(width=FALSE, height=FALSE)

menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="New", command=startnew)
filemenu.add_command(label="Save", command=savechat)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="Menu", menu=filemenu)

# Create Chat window
ChatBox = Text(root, bd=0, bg="#e5e5e5", height="8", width="50", font="Arial", )
ChatBox.config(state=DISABLED)
# Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="circle")
ChatBox['yscrollcommand'] = scrollbar.set
# Create Button to send message
SendButton = Button(root, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5,
                    bd=0, bg="white", activebackground="#3c9d9b", fg='#000000', command=send)
# Create the box to enter message
EntryBox = Text(root, bd=0, bg="#e5e5e5", width="12", height="5", font="Arial")
# EntryBox.bind("<Return>", send)

# Place all components on the screen
scrollbar.place(x=420, y=6, height=386)
ChatBox.place(x=7, y=7, height=386, width=400)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
root.config(menu=menubar)
root.mainloop()


# just an inspration on how to imply https://dzone.com/articles/python-chatbot-project-build-your-first-python-pro