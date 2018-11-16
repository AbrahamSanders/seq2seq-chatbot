"""
Reader class for generic CSV question-answer datasets
"""
from os import path
import pandas as pd

from dataset_readers.dataset_reader import DatasetReader

class CSVDatasetReader(DatasetReader):
    """Reader implementation for generic CSV question-answer datasets
    """
    def __init__(self):
        super(CSVDatasetReader, self).__init__("csv")
    
    def _get_dialog_lines_and_conversations(self, dataset_dir):
        """Get dialog lines and conversations. See base class for explanation.
        Args:
            See base class
        """
        csv_filepath = path.join(dataset_dir, "csv_data.csv")
        
        # Importing the dataset
        dataset = pd.read_csv(csv_filepath, dtype=str, na_filter=False)
        questions = dataset.iloc[:, 0].values
        answers = dataset.iloc[:, 1].values
        
        # Creating a dictionary that maps each line and its id
        conversations_ids = []
        id2line = {}
        for i in range(len(questions)):
            question = questions[i].strip()
            answer = answers[i].strip()
            if question != '' and answer != '':
                q_line_id = "{}_q".format(i)
                a_line_id = "{}_a".format(i)
                id2line[q_line_id] = question
                id2line[a_line_id] = answer
                conversations_ids.append([q_line_id, a_line_id])
        
        return id2line, conversations_ids