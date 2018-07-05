from gensim.models import KeyedVectors
import os
from configuration import Configurations
from glob import glob

config = Configurations()

class Word2VecTester(object):
    def __init__(self):
        self.model = KeyedVectors.load_word2vec_format(
            os.path.join(self.__find_directory(config.intent_home), 'word2vec_model.txt'),
            binary=False)

    def __find_directory(self, project_home):
        sub_directories = [os.path.join(project_home, d) for d in glob(os.path.join(project_home, '*'))]
        latest_folder = max(sub_directories, key=os.path.getmtime)
        return latest_folder


    def most_similar_words(self, word):
        return self.model.wv.most_similar(word)

    def get_similarity_between_two_words(self, word1, word2):
        print(self.model.wv.similarity(word1, word2))
        return self.model.wv.similarity(word1, word2)