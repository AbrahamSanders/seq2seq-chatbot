"""
General utility methods
"""
import os
import argparse
import datetime
from shutil import copyfile

from hparams import Hparams

def initialize_session(mode):
    """Helper method for initializing a chatbot training session 
    by loading the model dir from command line args and reading the hparams in

    Args:
        mode: "train" or "chat"
    """
    parser = argparse.ArgumentParser("Train a chatbot model" if mode == "train" else "Chat with a trained chatbot model")
    if mode == "train":
        ex_group = parser.add_mutually_exclusive_group(required=True)
        ex_group.add_argument("--datasetname", "-d", help="Subdirectory name within /datasets which contains the dataset. A new model will be trained using the dataset.")
        ex_group.add_argument("--checkpointfile", "-c", help="Path within /models structured as 'dataset_name/model_dir/checkpoint_name.ckpt'. Training will resume from the selected checkpoint. The checkpoint file should exist along with the hparams.json file.")
    elif mode == "chat":
        parser.add_argument("checkpointfile", help="Path within /models structured as 'dataset_name/model_dir/checkpoint_name.ckpt'. The checkpoint file should exist along with the hparams.json file and the vocabulary file(s).")
    else:
        raise ValueError("Unsupported session mode. Choose 'train' or 'chat'.")
    args = parser.parse_args()
    
    models_dir = os.path.join(os.getcwd(), "models")
    datasets_dir = os.path.join(os.getcwd(), "datasets")

    #Make sure script was run in the correct working directory
    if not os.path.isdir(models_dir) or not os.path.isdir(datasets_dir):
        raise NotADirectoryError("Cannot find models directory '/models' and datasets directory '/datasets' within working directory {0}. Make sure to set the working directory to the root Chatbot folder."
                                    .format(os.getcwd()))

    if mode == "train" and args.datasetname:
        dataset_name = args.datasetname
        model_dir = os.path.join(models_dir, dataset_name, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(model_dir, exist_ok=True)
        copyfile(os.path.join(os.getcwd(), "hparams.json"), os.path.join(model_dir, "hparams.json"))
        checkpoint = None
    elif args.checkpointfile:
        checkpoint = os.path.basename(args.checkpointfile)
        model_dir = os.path.dirname(args.checkpointfile)
        dataset_name = os.path.dirname(model_dir)
        if model_dir == "" or dataset_name == "":
            raise ValueError("--checkpointfile must specify a path within /models structured as 'dataset_name/model_dir/checkpoint_name.ckpt'.")
        model_dir = os.path.join(models_dir, model_dir)
        
        #Make sure checkpoint file & hparams file exists
        checkpoint_filepath = os.path.join(model_dir, checkpoint)
        checkpoint_hparams_filepath = os.path.join(model_dir, "hparams.json")
        if not os.path.isfile(checkpoint_filepath + ".meta"):
            raise FileNotFoundError("The checkpoint file '{0}' was not found.".format(checkpoint_filepath))
        if not os.path.isfile(checkpoint_hparams_filepath):
            raise FileNotFoundError("The hparams file '{0}' was not found. Make sure the checkpoint and hparams files are in the same directory"
                                    .format(checkpoint_hparams_filepath))
    else:
        raise ValueError("Invalid arguments. Use --help for proper usage.")

    #Load the hparams from file
    hparams_filepath = os.path.join(model_dir, "hparams.json")
    hparams = Hparams.load(hparams_filepath)
    dataset_dir = os.path.join(datasets_dir, dataset_name)

    return dataset_dir, model_dir, hparams, checkpoint