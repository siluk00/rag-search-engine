import string
from nltk.stem import PorterStemmer

def tokenize_input(words):
    stopwords, stemmer, table = __prebuild_tokenizetion()
    tokens = words.lower().translate(table).split() #uncapitalize and remove punctuation to list
    tokens = list(filter(lambda x: x != "",tokens)) #removes blank
    tokens = list(filter(lambda x: x not in stopwords, tokens)) #removes words without meaning
    tokens = list(map(lambda x: stemmer.stem(x),   tokens)) #turns words to their stem
    return tokens   

def tokenize_word(word):
    stopwords, stemmer, table = __prebuild_tokenizetion()
    token = word.lower().translate(table)
    if token in stopwords:
        return ""
    token = stemmer.stem(token)
    return token


def __prebuild_tokenizetion():
    stopwords = []
    with open("data/stopwords.txt", 'r') as f:
        txt = f.read()
        stopwords = txt.splitlines()
    
    stemmer = PorterStemmer()
    table = str.maketrans("", "", string.punctuation) #table of transformation, puntuation -> ""
    return stopwords, stemmer, table

