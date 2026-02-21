class InvertedIndex:
    def __init__(self, index, docmap):
        self.index = index
        self.docmap = docmap
    
    def _add_document(self, doc_id, text):
        tokens = text.split(" ")
        for token in tokens:
            self.index[token].add(doc_id)

    def get_document(self, term):
        document = []
        for elem in self.index[term]:
            document.append(elem)
        document.sort()
        return document
    
    def build(self):
        pass

