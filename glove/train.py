from gensim import utils
from gensim.scripts.glove2word2vec import glove2word2vec
from train_utils import SentencesLoader
from glove import Corpus, Glove

import tempfile
import logging
import os
import configparser

path_config_file = 'config.ini'
config = configparser.ConfigParser()
config.read(path_config_file)
models_dir = config['PATHS']['models_dir']
data_dir = config['PATHS']['data_dir']
log_dir = config['PATHS']['log_dir']
## Assumes training.txt already existing in log folder
log_file = os.path.join(log_dir, 'training.txt')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

paths = [  
  # Lemmatized w/ treetagger
  os.path.join(data_dir, 'Economics.txt'),
#   os.path.join(data_dir, 'Education_and_psychology.txt'),
#   os.path.join(data_dir, 'History.txt'),
#   os.path.join(data_dir, 'Law.txt'),
#   os.path.join(data_dir, 'Linguistics.txt'),
#   os.path.join(data_dir, 'Sociology.txt'),
#   os.path.join(data_dir, 'supercybercat.txt'),

]

CONTEXT_WINDOW = 10 # 5, 10
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
      glove.save(os.path.join(models_dir, model_name))

      glove_filename = os.path.join(models_dir, model_name)
      word2vec_output_file = glove_filename + '.word2vec'
      glove2word2vec(glove_filename, word2vec_output_file)

if __name__ == '__main__':
    main()