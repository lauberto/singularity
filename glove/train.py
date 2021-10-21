from gensim import utils
from gensim.scripts.glove2word2vec import glove2word2vec
from train_utils import SentencesLoader
from glove import Corpus, Glove

import tempfile
import logging
import os
import argparse

CONTEXT_WINDOW = 10 # 5, 10
SIZE = 500 # 200, 300, 500

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', type=str, help='Directory with corpus text files')
  parser.add_argument('--models-dir', type=str, help='Directory where to store trained Glove models')
  parser.add_argument('--size', default=500, type=int, help='Glove embedding size')
  parser.add_argument('--context-window', default=10, type=int, help='Context window for the Glove model to consider')
  args = parser.parser.parse_args()

  paths = [  
    # Lemmatized w/ treetagger
    os.path.join(args.data_dir, 'Economics.txt'),
    os.path.join(args.data_dir, 'Education_and_psychology.txt'),
    os.path.join(args.data_dir, 'History.txt'),
    os.path.join(args.data_dir, 'Law.txt'),
    os.path.join(args.data_dir, 'Linguistics.txt'),
    os.path.join(args.data_dir, 'Sociology.txt'),
    os.path.join(args.data_dir, 'supercybercat.txt'),
    ]

  for epochs in [30, 100]:
    for datapath in paths:
      print(f'Started training for: {datapath}', flush=True)
      model_name = os.path.splitext(os.path.basename(datapath))[0] + f'_{epochs}epx.model'
      sentences = SentencesLoader(datapath)
      corpus = Corpus()
      corpus.fit(sentences, window=CONTEXT_WINDOW)

      glove = Glove(no_components=SIZE, learning_rate=0.05) 
      glove.fit(corpus.matrix, epochs=epochs, no_threads=40, verbose=True)
      glove.add_dictionary(corpus.dictionary)
      glove.save(os.path.join(args.models_dir, model_name))

      glove_filename = os.path.join(args.models_dir, model_name)
      word2vec_output_file = glove_filename + '.word2vec'
      glove2word2vec(glove_filename, word2vec_output_file)

if __name__ == '__main__':
    main()