import json, string
from nltk.stem import PorterStemmer
from keyword_search_cli import tokenize_input

class InvertedIndex:
    def __init__(self, index, docmap):
        self.index = dict()
        self.docmap = dict()
    
    def _add_document(self, doc_id, text):
        if text in self.index:
            self.index[text].add(doc_id)
        else:
            self.index[text] = set()

    def get_document(self, term):
        document = []
        for elem in self.index[term]:
            document.append(elem)
        document.sort()
        return document
    
    def build(self):
        dict = {}
        with open("data/movies.json", 'r') as f:
                dict = json.load(f)
        results = []
    
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
                self._add_document(token, entry["id"])

            
             
             
             
             


