"""
Dataset generator implementation factory
"""
from os import path
from cornell_generator import CornellGenerator

def get_dataset_generator(dataset_dir):
    """Gets the appropriate generator implementation for the specified dataset name.

    Args:
        dataset_dir: The directory of the dataset to get a generator implementation for.
            The folder name of the Subdirectory within /datasets is the dataset name.
    """
    dataset_name = path.basename(dataset_dir)

    #When adding support for new datasets, add an instance of their generator class to the generaters array below.
    generators = [CornellGenerator()]

    for generator in generators:
        if generator.dataset_name == dataset_name:
            return generator

    raise ValueError("There is no dataset generator implementation for '{0}'. If this is a new dataset, please add one!".format(dataset_name))