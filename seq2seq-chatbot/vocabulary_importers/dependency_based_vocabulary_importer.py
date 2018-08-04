"""
Importer class for the Dependency-Based vocabulary (Levy & Goldberg, 2014)
https://levyomer.wordpress.com/2014/04/25/dependency-based-word-embeddings/
https://levyomer.files.wordpress.com/2014/04/dependency-based-word-embeddings-acl-2014.pdf
"""
from os import path

from vocabulary_importers.flatfile_vocabulary_importer import FlatFileVocabularyImporter
from vocabulary import Vocabulary

class DependencyBasedVocabularyImporter(FlatFileVocabularyImporter):
    """Importer implementation for the Dependency-Based vocabulary
    """
    def __init__(self):
        super(DependencyBasedVocabularyImporter, self).__init__("dependency_based", "deps.words", " ")
    
    def _process_token(self, token):
        """Perform token preprocessing (See base class for explanation)

        Args:
            See base class

        Returns:
            See base class
        """

        if token == "[":
            token = Vocabulary.SOS
        elif token == "]":
            token = Vocabulary.EOS
        elif token == "iz":
            token = Vocabulary.OUT
        elif token == "--":
            token = Vocabulary.PAD
        elif token == "''":
            token = "?"

        return token