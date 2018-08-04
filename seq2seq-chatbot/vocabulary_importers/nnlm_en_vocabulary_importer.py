"""
Importer class for the nnlm english vocabulary (Bengio et al, 2003)
https://www.tensorflow.org/hub/modules/google/nnlm-en-dim128/1
http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
"""
from os import path

from vocabulary_importers.checkpoint_vocabulary_importer import CheckpointVocabularyImporter
from vocabulary import Vocabulary

class NnlmEnVocabularyImporter(CheckpointVocabularyImporter):
    """Importer implementation for the nnlm english vocabulary
    """
    def __init__(self):
        super(NnlmEnVocabularyImporter, self).__init__("nnlm_en", "tokens.txt", "embeddings")
    
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
        elif token == "--":
            token = Vocabulary.PAD

        return token