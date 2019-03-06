"""
Reader class for the DailyDialog dataset
"""
from os import path

from dataset_readers.dataset_reader import DatasetReader

class DailyDialogDatasetReader(DatasetReader):
    """Reader implementation for the DailyDialog dataset
    """
    def __init__(self):
        super(DailyDialogDatasetReader, self).__init__("dailydialog")
    
    def _get_dialog_lines_and_conversations(self, dataset_dir):
        """Get dialog lines and conversations. See base class for explanation.

        Args:
            See base class
        """
        dialogues_filepath = path.join(dataset_dir, "dialogues_text.txt")
        
        # Importing the dataset
        with open(dialogues_filepath, encoding="utf-8", errors="ignore") as file:
            conversations = file.read().split("\n")

        # Reading in the dataset
        id2line = {}
        conversations_ids = []

        for i in range(len(conversations)):
            turns = conversations[i].split("__eou__")[:-1]
            conv_ids = []
            for j in range(len(turns)):
                turn = turns[j]
                
                turn = turn.replace("’", "'")
                turn = turn.replace(" ' ", "'")
                turn = turn.replace(" ? ", "? ")
                turn = turn.replace(" ... ",  "... ")
                turn = turn.replace(" .. . ",  "... ")
                turn = turn.replace(" .. ",  ".. ")
                turn = turn.replace(" . ",  ". ")
                turn = turn.replace(" ! ", "! ")
                turn = turn.replace(" , ", ", ")
                turn = turn.replace(" $ ", " $")
                turn = turn.replace(" % ", "% ")
                turn = turn.replace(" # ", " #")
                turn = turn.replace(" ( ", " (")
                turn = turn.replace(" ) ", ") ")
                turn = turn.replace(" / ", "/")
                turn = turn.replace("\\", "")
                turn = turn.strip()
                
                turn_id = "C{0}_T{1}".format(i, j)
                conv_ids.append(turn_id)
                id2line[turn_id] = turn
            conversations_ids.append(conv_ids)
        
        return id2line, conversations_ids
    