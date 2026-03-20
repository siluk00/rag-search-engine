import string
from nltk.stem import PorterStemmer
import json, os, pickle, collections, math
from constants import BM25_K1, BM25_B

def tokenize_input(words) -> list[str]:
    stopwords, stemmer, table = __prebuild_tokenizetion()
    tokens = words.lower().translate(table).split() #uncapitalize and remove punctuation to list
    tokens = list(filter(lambda x: x != "",tokens)) #removes blank
    tokens = list(filter(lambda x: x not in stopwords, tokens)) #removes words without meaning
    tokens = list(map(lambda x: stemmer.stem(x),   tokens)) #turns words to their stem
    return tokens   

def tokenize_word(word: str) -> str:
    tokens = tokenize_input(word)
    if len(tokens) == 1:
        token = tokens[0] 
    else:
        print("invalid token")
        exit(1)
    return token

def __prebuild_tokenizetion():
    stopwords = []
    with open("data/stopwords.txt", 'r') as f:
        txt = f.read()
        stopwords = txt.splitlines()
    
    stemmer = PorterStemmer()
    table = str.maketrans("", "", string.punctuation) #table of transformation, puntuation -> ""
    return stopwords, stemmer, table

class InvertedIndex:
    CACHE_DIR = "cache"

    def __init__(self):
        os.makedirs(self.CACHE_DIR, exist_ok=True)

        self.index = dict() #a dictionary mappin words to ids / inverted index
        self.docmap = dict() # a dictionary mapping id to full document
        self.term_frequencies = dict() # a dictionary mapping doc_id to Counter object 
        self.doc_lengths = dict() #a dictionary mapping doc id to amount of words

        self.index_path = f"{self.CACHE_DIR}/index.pkl"
        self.docmap_path = f"{self.CACHE_DIR}/docmap.pkl"
        self.term_frequencies_path = f"{self.CACHE_DIR}/term_frequencies.pkl"
        self.doc_lengths_path = f"{self.CACHE_DIR}/doc_lengths.pkl"

    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        return sum(self.doc_lengths.values())/len(self.doc_lengths)
    
    #adds text to the set with doc_id key
    def __add_document(self, doc_id: int, text: str):
        tokens = tokenize_input(text)
        self.doc_lengths[doc_id] = len(tokens)

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
        return math.log((N-df+0.5)/(df+0.5)+1) #idf bm25 formula
    
    #calculates bm25_tf for given term on document with id doc_id
    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1, b = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id]/self.__get_avg_doc_length()) #normalization of length
        return (tf * (k1 + 1)/(tf+k1 * length_norm)) #tf saturation formula
    
    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_idf(term) * self.get_bm25_tf(doc_id, term)
    
    def bm25_search(self, query, limit):
        tokens = tokenize_input(query)
        scores = dict()

        for doc_id in self.docmap.keys():
            total_score = 0
            for token in tokens:
                total_score += self.bm25(doc_id, token)
            scores[doc_id] = total_score
        
        sorted_list = sorted(scores, key=scores.get, reverse=True)[:limit]
        for i, doc_id in enumerate(sorted_list):
            print(f"{i+1}. ({doc_id}) {self.docmap[doc_id]["title"]} - Score: {scores[doc_id]:.2f}")

    #builds the inverted index 
    def build(self):
        dict = {}
        with open("data/movies.json", 'r') as f:
                dict = json.load(f)

        for entry in dict["movies"]:
            self.__add_document(entry["id"], f"{entry["title"]} {entry["description"]}")
            self.docmap[entry["id"]] = entry
    
    def save(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        
        with open(self.term_frequencies_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)

        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
        
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
       
        with open(self.term_frequencies_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)

        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)
        

            
             
             
             
             




