"""
Reader class for the Cornell movie dialog dataset
"""
from os import path

from dataset_readers.dataset_reader import DatasetReader

class CornellDatasetReader(DatasetReader):
    """Reader implementation for the Cornell movie dialog dataset
    """
    def __init__(self):
        super(CornellDatasetReader, self).__init__("cornell_movie_dialog")
    
    def _get_dialog_lines_and_conversations(self, dataset_dir):
        """Get dialog lines and conversations. See base class for explanation.

        Args:
            See base class
        """
        movie_lines_filepath = path.join(dataset_dir, "movie_lines.txt")
        movie_conversations_filepath = path.join(dataset_dir, "movie_conversations.txt")
        
        # Importing the dataset
        with open(movie_lines_filepath, encoding="utf-8", errors="ignore") as file:
            lines = file.read().split("\n")
        
        with open(movie_conversations_filepath, encoding="utf-8", errors="ignore") as file:
            conversations = file.read().split("\n")
        
        # Creating a dictionary that maps each line and its id
        id2line = {}
        for line in lines:
            _line = line.split(" +++$+++ ")
            if len(_line) == 5:
                id2line[_line[0]] = _line[4]
        
        # Creating a list of all of the conversations
        conversations_ids = []
        for conversation in conversations[:-1]:
            _conversation = conversation.split(" +++$+++ ")[-1][1:-1].replace("'", "").replace(" ", "")
            conv_ids = _conversation.split(",")
            conversations_ids.append(conv_ids)
        
        return id2line, conversations_ids
    