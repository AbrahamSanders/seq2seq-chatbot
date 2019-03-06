"""
Dataset reader implementation factory
"""
from os import path
from dataset_readers.cornell_dataset_reader import CornellDatasetReader
from dataset_readers.csv_dataset_reader import CSVDatasetReader
from dataset_readers.dailydialog_dataset_reader import DailyDialogDatasetReader

def get_dataset_reader(dataset_dir):
    """Gets the appropriate reader implementation for the specified dataset name.

    Args:
        dataset_dir: The directory of the dataset to get a reader implementation for.
    """
    dataset_name = path.basename(dataset_dir)

    #When adding support for new datasets, add an instance of their reader class to the reader array below.
    readers = [CornellDatasetReader(), CSVDatasetReader(), DailyDialogDatasetReader()]

    for reader in readers:
        if reader.dataset_name == dataset_name:
            return reader

    raise ValueError("There is no dataset reader implementation for '{0}'. If this is a new dataset, please add one!".format(dataset_name))
