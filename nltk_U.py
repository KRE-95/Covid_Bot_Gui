import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

"""
Here I am going  use an framework called NLTK ( natural language toolkit)
# This is a Python library to work with human language.
# these concepts from this toolkit used tokenization and stemming and bag.

"""


def tokenize(sentence):
    """
      we apply tokenization, so  split our sentence into the different words and here also the
      punctuation, character ( from jason file)
      we lower all the words, so is with a capital I becomes i with the lower

    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    Then we stem the words - the ending get chopped off then we also exclude, punctuation characters so we don't need
    the question marks or the exclamation marks and then based on this array we calculate the so called bag of word.

    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    This func helps the bot the detect the pattern on the given sentence by the user. By scanning the
    zeros and one's . 1 is written when the sentence word matches, while 0 is not.

    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    Do you sell N95 mask?
    sentence = ["Do", "you", "sell", "N95", "mask" ]
    words = ["hi", "Do", "I", "you", "sell", "thank", "N95" ,  "mask", "anticbac" ]
    bog   = [  0 ,  1 ,  0 ,   1 ,    1 ,    0 ,        1    ,  1  ,     0   ]

    What is even more impressive, you can make the bot to recognize the patterns of words in the sentence.
    Even if the sentence is not written inside the jason.file e.g. tag items.

    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)  # datatype 32, document. numpy doc
    for idx, w in enumerate(words):  # idx is index , w for word
        if w in sentence_words:  # if the word is the inside the sentence_words then
            bag[idx] = 1  # read is as 1  if it exist.

    return bag

# https://numpy.org/doc/stable/user/basics.types.html
