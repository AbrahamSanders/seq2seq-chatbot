"""
General utility methods
"""
import os
import argparse
import datetime
import platform
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
        ex_group.add_argument("--datasetdir", "-d", help="Path structured as datasets/dataset_name. A new model will be trained using the dataset contained in this directory.")
        ex_group.add_argument("--checkpointfile", "-c", help="Path structured as 'models/dataset_name/model_name/checkpoint_name.ckpt'. Training will resume from the selected checkpoint. The hparams.json file should exist in the same directory as the checkpoint.")
        em_group = parser.add_argument_group()
        em_group.add_argument("--encoderembeddingsdir", "--embeddingsdir", "-e", help="Path structured as embeddings/embeddings_name. Encoder (& Decoder if shared) vocabulary and embeddings will be initialized from the checkpoint file and tokens file contained in this directory.")
        em_group.add_argument("--decoderembeddingsdir", help="Path structured as embeddings/embeddings_name. Decoder vocabulary and embeddings will be initialized from the checkpoint file and tokens file contained in this directory.")
    elif mode == "chat":
        parser.add_argument("checkpointfile", help="Path structured as 'models/dataset_name/model_name/checkpoint_name.ckpt'. The hparams.json file and the vocabulary file(s) should exist in the same directory as the checkpoint.")
    else:
        raise ValueError("Unsupported session mode. Choose 'train' or 'chat'.")
    args = parser.parse_args()
    
    #Make sure script was run in the correct working directory
    models_dir = "models"
    datasets_dir = "datasets"
    if not os.path.isdir(models_dir) or not os.path.isdir(datasets_dir):
        raise NotADirectoryError("Cannot find models directory 'models' and datasets directory 'datasets' within working directory '{0}'. Make sure to set the working directory to the chatbot root folder."
                                    .format(os.getcwd()))

    encoder_embeddings_dir = decoder_embeddings_dir = None
    if mode == "train":
        #If provided, make sure the embeddings exist
        if args.encoderembeddingsdir:
            encoder_embeddings_dir = os.path.relpath(args.encoderembeddingsdir)
            if not os.path.isdir(encoder_embeddings_dir):
                raise NotADirectoryError("Cannot find embeddings directory '{0}'".format(os.path.realpath(encoder_embeddings_dir)))
        if args.decoderembeddingsdir:
            decoder_embeddings_dir = os.path.relpath(args.decoderembeddingsdir)
            if not os.path.isdir(decoder_embeddings_dir):
                raise NotADirectoryError("Cannot find embeddings directory '{0}'".format(os.path.realpath(decoder_embeddings_dir)))
                
    if mode == "train" and args.datasetdir:
        #Make sure dataset exists
        dataset_dir = os.path.relpath(args.datasetdir)
        if not os.path.isdir(dataset_dir):
            raise NotADirectoryError("Cannot find dataset directory '{0}'".format(os.path.realpath(dataset_dir)))
        #Create the new model directory
        dataset_name = os.path.basename(dataset_dir)
        model_dir = os.path.join("models", dataset_name, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(model_dir, exist_ok=True)
        copyfile("hparams.json", os.path.join(model_dir, "hparams.json"))
        checkpoint = None
    elif args.checkpointfile:
        #Make sure checkpoint file & hparams file exists
        checkpoint_filepath = os.path.relpath(args.checkpointfile)
        if not os.path.isfile(checkpoint_filepath + ".meta"):
            raise FileNotFoundError("The checkpoint file '{0}' was not found.".format(os.path.realpath(checkpoint_filepath)))
        #Get the checkpoint model directory
        checkpoint = os.path.basename(checkpoint_filepath)
        model_dir = os.path.dirname(checkpoint_filepath)
        dataset_name = os.path.basename(os.path.dirname(model_dir))
        dataset_dir = os.path.join(datasets_dir, dataset_name)
    else:
        raise ValueError("Invalid arguments. Use --help for proper usage.")

    #Load the hparams from file
    hparams_filepath = os.path.join(model_dir, "hparams.json")
    hparams = Hparams.load(hparams_filepath)

    return dataset_dir, model_dir, hparams, checkpoint, encoder_embeddings_dir, decoder_embeddings_dir

def initialize_session_server(checkpointfile):
    #Make sure checkpoint file & hparams file exists
    checkpoint_filepath = os.path.relpath(checkpointfile)
    if not os.path.isfile(checkpoint_filepath + ".meta"):
        raise FileNotFoundError("The checkpoint file '{0}' was not found.".format(os.path.realpath(checkpoint_filepath)))
    #Get the checkpoint model directory
    checkpoint = os.path.basename(checkpoint_filepath)
    model_dir = os.path.dirname(checkpoint_filepath)

    #Load the hparams from file
    hparams_filepath = os.path.join(model_dir, "hparams.json")
    hparams = Hparams.load(hparams_filepath)

    return model_dir, hparams, checkpoint

def create_batch_files(model_dir, checkpoint_training, checkpoint_val, encoder_embeddings_dir, decoder_embeddings_dir):
    os_type = platform.system().lower()
    if os_type == "windows":
        if checkpoint_training is not None:
            create_windows_batch_files(model_dir, checkpoint_training, encoder_embeddings_dir, decoder_embeddings_dir)
        if checkpoint_val is not None:
            create_windows_batch_files(model_dir, checkpoint_val, encoder_embeddings_dir, decoder_embeddings_dir)
    elif os_type == "darwin":
        pass
    elif os_type == "linux":
        pass
    else:
        pass

def create_windows_batch_files(model_dir, checkpoint, encoder_embeddings_dir, decoder_embeddings_dir):
    if "CONDA_PREFIX" in os.environ:
        conda_prefix = os.environ["CONDA_PREFIX"]
        conda_activate = os.path.join(conda_prefix, r"scripts\activate.bat")
        checkpoint_file = os.path.join(model_dir, checkpoint)
        checkpoint_name = os.path.splitext(checkpoint)[0]

        #Resume training batch file
        batch_file = os.path.join(model_dir, "resume_training_{0}.bat".format(checkpoint_name))
        with open(batch_file, mode="w", encoding="utf-8") as file:
            file.write("\n".join([
                                    "call {0} {1}".format(conda_activate, conda_prefix),
                                    r"cd ..\..\..",
                                    "python train.py --checkpointfile=\"{0}\"{1}{2}".format(checkpoint_file,
                                                    " --encoderembeddingsdir={0}".format(encoder_embeddings_dir) if encoder_embeddings_dir is not None else "",
                                                    " --decoderembeddingsdir={0}".format(decoder_embeddings_dir) if decoder_embeddings_dir is not None else ""),
                                    "",
                                    "cmd /k"
                                ]))

        #Chat batch file
        batch_file = os.path.join(model_dir, "chat_console_{0}.bat".format(checkpoint_name))
        with open(batch_file, mode="w", encoding="utf-8") as file:
            file.write("\n".join([
                                    "call {0} {1}".format(conda_activate, conda_prefix),
                                    r"cd ..\..\..",
                                    "python chat.py \"{0}\"".format(checkpoint_file),
                                    "",
                                    "cmd /k"
                                ]))
        
        #Chat web batch file
        batch_file = os.path.join(model_dir, "chat_web_{0}.bat".format(checkpoint_name))
        with open(batch_file, mode="w", encoding="utf-8") as file:
            file.write("\n".join([
                                    "call {0} {1}".format(conda_activate, conda_prefix),
                                    r"cd ..\..\..",
                                    "set FLASK_APP=chat_web.py",
                                    "flask serve_chat \"{0}\" -p 8080".format(checkpoint_file),
                                    "",
                                    "cmd /k"
                                ]))

        #Tensorboard batch file
        batch_file = os.path.join(model_dir, "tensorboard_{0}.bat".format(checkpoint_name))
        with open(batch_file, mode="w", encoding="utf-8") as file:
            file.write("\n".join([
                                    "call {0} {1}".format(conda_activate, conda_prefix),
                                    "tensorboard --logdir=."
                                ]))
