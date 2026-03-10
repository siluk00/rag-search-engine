import json, string, os, pickle
from nltk.stem import PorterStemmer

def tokenize_input(phrase, stopwords, table, stemmer): #returns a list of tokenized inputs
    tokens = phrase.lower().translate(table).split(" ") #uncapitalize and remove punctuation to list
    tokens = list(filter(lambda x: x != "",tokens)) #removes blank
    tokens = list(filter(lambda x: x not in stopwords, tokens)) #removes words without meaning
    tokens = list(map(lambda x: stemmer.stem(x),   tokens)) #turns words to their stem
    return tokens 

class InvertedIndex:
    def __init__(self):
        self.index = dict() #a dictionary mappin words to id / inverted index
        self.docmap = dict() # a dictionary mapping id to  title + description
    
    #adds text to the set with doc_id key
    def _add_document(self, doc_id, text):
        if not (text in self.index):
            self.index[text] = set()
        self.index[text].add(doc_id)

    # gets the set of doc_ids as  list for the term searched
    def get_document(self, term):
        if term not in self.index:
            return []
        
        document = []
        document.extend(self.index[term]) #extend transforms set into list
        document.sort()
        return document #returns a list
    
    #builds the inverted index 
    def build(self):
        dict = {}
        with open("data/movies.json", 'r') as f:
                dict = json.load(f)
    
        stopwords = []
        with open("data/stopwords.txt", 'r') as f:
            txt = f.read()
            stopwords = txt.splitlines()
    
        stemmer = PorterStemmer()
        table = str.maketrans("", "", string.punctuation)

        for entry in dict["movies"]:
            tokens = tokenize_input(f"{entry["title"]} {entry["description"]}", stopwords, table, stemmer)
            self.docmap[entry["id"]] = entry

            for token in tokens:
                self._add_document(entry["id"], token)
    
    def save(self):
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        with open(index_path, 'wb') as f:
            pickle.dump(self.index, f)
        
        with open(docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)

    def load(self):
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"
        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)
        with open(docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
       
        

            
             
             
             
             


