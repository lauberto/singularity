from gensim.models.callbacks import CallbackAny2Vec
from gensim import utils

class SentencesLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def __iter__(self):
        for line in open(self.data_path, "r", encoding="utf-8"):
            yield utils.simple_preprocess(line)