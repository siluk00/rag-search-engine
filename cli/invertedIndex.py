import json, os, pickle, collections, math
from keyword_search import tokenize_input, tokenize_word



class InvertedIndex:
    def __init__(self):
        self.index = dict() #a dictionary mappin words to ids / inverted index
        self.docmap = dict() # a dictionary mapping id to  title + description
        self.term_frequencies = dict() # a dictionary mapping doc_id to Counter object 
    
    #adds text to the set with doc_id key
    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_input(text)

        for token in tokens:
            if not (token in self.index):
                self.index[token] = set()
            if not(doc_id in self.term_frequencies):
                self.term_frequencies[doc_id] = collections.Counter()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    # gets the set of doc_ids as  list for the term searched
    def get_document(self, term:str) -> list[int]:
        if term not in self.index:
            return []
        
        document = []
        document.extend(self.index[term]) #extend transforms set into list
        document.sort()
        return document #returns a list
    
    def get_tf(self, doc_id: int, term: str) -> int:
        token = tokenize_word(term)
        if token == "":
            return 0
        if token not in self.term_frequencies[doc_id]:
            return 0
        if doc_id not in self.term_frequencies:
            print("doc_id doesn'd exit")
            exit(1)
        return self.term_frequencies[doc_id][token]

    #calculates bm25_idf for given term    
    def get_bm25_idf(self, term: str) -> float:
        N = len(self.docmap)
        token = tokenize_word(term)
        
        df = len(self.get_document(token))
        return math.log((N-df+0.5)/(df+0.5)+1)
    
    #builds the inverted index 
    def build(self):
        dict = {}
        with open("data/movies.json", 'r') as f:
                dict = json.load(f)

        for entry in dict["movies"]:
            self.__add_document(entry["id"], f"{entry["title"]} {entry["description"]}")
            self.docmap[entry["id"]] = entry
    
    def save(self):
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"
        term_frequencies_path = "cache/term_frequencies.pkl"

        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        with open(index_path, 'wb') as f:
            pickle.dump(self.index, f)
        
        with open(docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        
        with open(term_frequencies_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        index_path = "cache/index.pkl"
        docmap_path = "cache/docmap.pkl"
        term_frequencies_path = "cache/term_frequencies.pkl"

        with open(index_path, 'rb') as f:
            self.index = pickle.load(f)
        
        with open(docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
       
        with open(term_frequencies_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)
        

            
             
             
             
             


