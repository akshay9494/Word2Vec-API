import os
import gensim
import glob
from tika import parser
from configuration import Configurations
from datetime import datetime
from gensim.models import KeyedVectors

config = Configurations()


class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in glob.glob(os.path.join(self.dirname, '*')):
            parsed = parser.from_file(fname)
            for line in parsed['content'].split('\n'):
                yield gensim.utils.simple_preprocess(line)


class Word2VecTrainer(object):
    def __init__(self):
        self.train_folder_name = 'embeddings_train_{}'.format(datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
        self.__pre_train_essentials()

    def __pre_train_essentials(self):
        self.train_folder_path = os.path.join(config.intent_home, self.train_folder_name)
        if not os.path.isdir(self.train_folder_path):
            os.makedirs(self.train_folder_path)

    def train_from_scratch(self):
        sentences = MySentences(config.data_source_properties.file_properties.folder_path)

        bigram_transformer = gensim.models.Phrases(sentences)

        model = gensim.models.Word2Vec(bigram_transformer[sentences], min_count=10, compute_loss=True)
        print('Training complete!')

        model.wv.save_word2vec_format(os.path.join(self.train_folder_path, 'word2vec_model.bin'), binary=True)
        model.save(os.path.join(self.train_folder_path, 'word2vec_model'))

        training_statistics = {}

        training_statistics['training_loss'] = model.get_latest_training_loss()
        training_statistics['vocab_length'] = len(list(model.wv.vocab))

        return training_statistics

    def train_on_existing(self):
        sentences = MySentences(config.data_source_properties.file_properties.folder_path)

        bigram_transformer = gensim.models.Phrases(sentences)

        # checks to be added for embeddings path
        model = gensim.models.Word2Vec.load(config.data_source_properties.existing_embedding_path)

        model.train(bigram_transformer[sentences],
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

        print('Training complete!')

        model.wv.save_word2vec_format(os.path.join(self.train_folder_path, 'word2vec_model.txt'), binary=False)

        training_statistics = {}

        training_statistics['training_loss'] = model.get_latest_training_loss()
        training_statistics['vocab_length'] = len(list(model.wv.vocab))

        return training_statistics


