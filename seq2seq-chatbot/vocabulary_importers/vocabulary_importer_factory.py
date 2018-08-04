"""
Vocabulary importer implementation factory
"""
from os import path
from vocabulary_importers.word2vec_wikipedia_vocabulary_importer import Word2vecWikipediaVocabularyImporter
from vocabulary_importers.nnlm_en_vocabulary_importer import NnlmEnVocabularyImporter
from vocabulary_importers.dependency_based_vocabulary_importer import DependencyBasedVocabularyImporter

def get_vocabulary_importer(vocabulary_dir):
    """Gets the appropriate importer implementation for the specified vocabulary name.

    Args:
        vocabulary_dir: The directory of the vocabulary to get a importer implementation for.
    """
    vocabulary_name = path.basename(vocabulary_dir)

    #When adding support for new vocabularies, add an instance of their importer class to the importer array below.
    importers = [Word2vecWikipediaVocabularyImporter(),
                 NnlmEnVocabularyImporter(),
                 DependencyBasedVocabularyImporter()]

    for importer in importers:
        if importer.vocabulary_name == vocabulary_name:
            return importer

    raise ValueError("There is no vocabulary importer implementation for '{0}'. If this is a new vocabulary, please add one!".format(vocabulary_name))