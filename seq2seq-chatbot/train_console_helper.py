"""
Console helper for training session
"""

def write_vocabulary_import_stats(vocabulary_import_stats):
    print("  Stats:")
    print("      External vocab size: {0}".format(vocabulary_import_stats.external_vocabulary_size))
    if vocabulary_import_stats.dataset_vocabulary_size is not None:
        print("      Dataset vocab size: {0}".format(vocabulary_import_stats.dataset_vocabulary_size))
    if vocabulary_import_stats.intersection_size is not None:
        print("      Intersection size: {0}".format(vocabulary_import_stats.intersection_size))