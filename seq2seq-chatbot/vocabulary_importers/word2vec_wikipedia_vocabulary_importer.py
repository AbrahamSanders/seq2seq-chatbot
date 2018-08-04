"""
Importer class for the word2vec wikipedia vocabulary (Mikolov et al, 2013)
https://www.tensorflow.org/hub/modules/google/Wiki-words-250/1
https://arxiv.org/abs/1301.3781
"""
from os import path

from vocabulary_importers.checkpoint_vocabulary_importer import CheckpointVocabularyImporter
from vocabulary import Vocabulary

class Word2vecWikipediaVocabularyImporter(CheckpointVocabularyImporter):
    """Importer implementation for the word2vec wikipedia vocabulary
    """
    def __init__(self):
        super(Word2vecWikipediaVocabularyImporter, self).__init__("word2vec_wikipedia", "tokens.txt", "embeddings")
    
    def _process_token(self, token):
        """Perform token preprocessing (See base class for explanation)

        Args:
            See base class

        Returns:
            See base class
        """

        if token == "<S>":
            token = Vocabulary.SOS
        elif token == "</S>":
            token = Vocabulary.EOS
        elif token == "<UNK>":
            token = Vocabulary.OUT
        elif token == "#!#":
            token = "!"
        elif token == "#.#":
            token = "."
        elif token == "#?#":
            token = "?"

        return token