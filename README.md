# seq2seq-chatbot
A sequence2sequence chatbot implementation with TensorFlow.

## Chatting with a trained model
To chat with a trained model from a python console:

1. Set console working directory to the **seq2seq-chatbot** directory. This directory should have the **models** and **datasets** directories directly within it.

2. Run chat.py with the model checkpoint path:
```shell
run chat.py models\dataset_name\model_name\checkpoint.ckpt
```

For example, to chat with the trained cornell movie dialog model **trained_model_v1**:

1. Download and unzip [trained_model_v1](seq2seq-chatbot/models/cornell_movie_dialog/README.md) into the [seq2seq-chatbot/models/cornell_movie_dialog](seq2seq-chatbot/models/cornell_movie_dialog) folder

2. Set console working directory to the **seq2seq-chatbot** directory

3. Run:
```shell
run chat.py models\cornell_movie_dialog\trained_model_v1\best_weights_training.ckpt
```

The result should look like this:

![chat](doc_files/chat.png "chat")

## Training a model
To train a model from a python console:

1. Configure the [hparams.json](seq2seq-chatbot/hparams.json) file to the desired training hyperparameters

2. Set console working directory to the **seq2seq-chatbot** directory. This directory should have the **models** and **datasets** directories directly within it.

3. To train a new model, run train.py with the dataset path:
```shell
run train.py --datasetdir=datasets\dataset_name
```

Or to resume training an existing model, run train.py with the model checkpoint path:
```shell
run train.py --checkpointfile=models\dataset_name\model_name\checkpoint.ckpt
```

For example, to train a new model on the cornell movie dialog dataset with default hyperparameters:

1. Set console working directory to the **seq2seq-chatbot** directory

2. Run:
```shell
run train.py --datasetdir=datasets\cornell_movie_dialog
```

The result should look like this:

![train](doc_files/train.png "train")

## Visualizing a model in TensorBoard
[TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) is a great tool for visualizing what is going on under the hood when a TensorFlow model is being trained.

To start TensorBoard from a terminal:
```shell
tensorboard --logdir=model_dir
```

Where model_dir is the path to the directory where the model checkpoint file is. For example, to view the trained cornell movie dialog model **trained_model_v1**:
```shell
tensorboard --logdir=models\cornell_movie_dialog\trained_model_v1
```

### Visualize Training
Coming soon...

### Visualize model graph
Coming soon...

### Visualize word embeddings
TensorBoard can project the word embeddings into 3D space by performing a dimensionality reduction technique like PCA or T-SNE, and can allow you to explore how your model has grouped together the words in your vocabulary by viewing nearest neighbors in the embedding space for any word.
More about word embeddings in TensorFlow and the TensorBoard projector can be found [here](https://www.tensorflow.org/programmers_guide/embedding).

When launching TensorBoard for a model directory and selecting the "Projector" tab, it should look like this:
![train](doc_files/tensorboard_projector.png "train")

## Adding a new dataset
Instructions coming soon...

## Dependencies
The following python packages are used in seq2seq-chatbot:
(excluding packages that come with Anaconda)

- [TensorFlow](https://www.tensorflow.org/)
    ```shell
    pip install --upgrade tensorflow
    ```
    For GPU support: [(See here for full GPU install instructions including CUDA and cuDNN)](https://www.tensorflow.org/install/)
    ```shell
    pip install --upgrade tensorflow-gpu
    ```

- [jsonpickle](https://jsonpickle.github.io/)
    ```shell
    pip install --upgrade jsonpickle
    ```

## Roadmap
- Train a chatbot model using pre-trained word embeddings such as word2vec or GloVe.
- Extend the model with a binary classifier that can predict when a change in topic is occurring during a conversation. This would allow the chatbot to throttle use of dialog context (prepended conversation history) more intelligently.
- Implement an online learning mechanism and persistent storage so the bot can update its training dataset and incorporate newly learned facts on the fly.
- Create an Alexa skill for the bot :-)
  
## Acknowledgements
This implementation was inspired by:
- Kirill Eremenko & Hadelin de Ponteves [Deep NLP Udemy course](https://www.udemy.com/chatbot/)
- TensorFlow's [Neural Machine Translation (seq2seq) Tutorial](https://www.tensorflow.org/tutorials/seq2seq)
  - [TF NMT GitHub](https://github.com/tensorflow/nmt)
  
