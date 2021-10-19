from gensim import utils
from gensim.scripts.glove2word2vec import glove2word2vec
from train_utils import SentencesLoader
from glove_binary import Corpus, Glove

import tempfile
import logging
import os
import configparser

path_current_directory = os.path.dirname(__file__)
path_config_file = os.path.join(path_current_directory, 'config.ini')
config = configparser.ConfigParser()
config.read(path_config_file)
models_dir = config['PATHS']['models_dir']
data_dir = config['PATHS']['data_dir']
lemmas_dir = 'preprocessed/full_domains/lemmas/'
log_dir = config['PATHS']['log_dir']
log_file = os.path.join(log_dir, 'training.txt')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

paths = [  
  # Lemmatized w/ treetagger
  os.path.join(data_dir, lemmas_dir, 'Economics.txt'),
#   os.path.join(data_dir, lemmas_dir, 'Education_and_psychology.txt'),
#   os.path.join(data_dir, lemmas_dir, 'History.txt'),
#   os.path.join(data_dir, lemmas_dir, 'Law.txt'),
  os.path.join(data_dir, lemmas_dir, 'Linguistics.txt'),
#   os.path.join(data_dir, lemmas_dir, 'Sociology.txt'),
  os.path.join(data_dir, lemmas_dir, 'supercybercat.txt'),

  # Lemmatized w/ UDPipe
]

CONTEXT_WINDOW = 10 # 5, 10
# MIN_COUNT = 5 note: in future, figure out how to set min_count for GloVe via this library
EPOCHS = 30
SIZE = 500 # 200, 300, 500

def main():
  for epochs in [30, 100]:
    for datapath in paths:
      print(f'Started training for: {datapath}', flush=True)
      model_name = os.path.splitext(os.path.basename(datapath))[0] + f'_{epochs}epx.model'
      sentences = SentencesLoader(datapath)
      corpus = Corpus()
      corpus.fit(sentences, window=CONTEXT_WINDOW)

      glove = Glove(no_components=SIZE, learning_rate=0.05) 
      glove.fit(corpus.matrix, epochs=EPOCHS, no_threads=40, verbose=True)
      glove.add_dictionary(corpus.dictionary)
      glove.save(os.path.join(models_dir, 'glove', model_name))

      glove_filename = os.path.join(models_dir, 'glove', model_name)
      word2vec_output_file = glove_filename + '.word2vec'
      glove2word2vec(glove_filename, word2vec_output_file)

if __name__ == '__main__':
    main()