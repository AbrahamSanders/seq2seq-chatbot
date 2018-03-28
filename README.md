# seq2seq-chatbot
A sequence2sequence chatbot implementation with TensorFlow.

## Chatting with a trained model
To chat with a trained model from a python console:

1. Set console working directory to the **/seq2seq-chatbot** directory. This directory should have the **/models** and **/datasets** directories directly within it.

2. Run chat.py with the model checkpoint path:
```shell
run chat.py dataset_name\model_name\checkpoint.ckpt
```

**Note:** The path **dataset_name\model_name\checkpoint.ckpt** is relative to the **\seq2seq-chatbot\models** directory. So the full path to the model checkpoint would be : **$Repo\seq2seq-chatbot\models\dataset_name\model_name\checkpoint.ckpt**

For example, to chat with the trained cornell movie dialog model:

1. Unzip the cornell movie dialog dataset (cornell_movie_dialog.zip) into the [/seq2seq-chatbot/datasets/cornell_movie_dialog](/seq2seq-chatbot/datasets/cornell_movie_dialog) folder

2. Download and unzip [trained_model_v1](/seq2seq-chatbot/models/cornell_movie_dialog/README.md) into the [/seq2seq-chatbot/models/cornell_movie_dialog](/seq2seq-chatbot/models/cornell_movie_dialog) folder

3. Run:
```shell
run chat.py cornell_movie_dialog\trained_model_v1\best_weights_training.ckpt
```

The result should look like this:

![chat](/doc_files/chat.png "chat")

## Training a model
Instructions coming soon...

## Visualizing a model in TensorBoard
Instructions coming soon...

### Visualize Training
Coming soon...

### Visualize model graph
Coming soon...

### Visualize word embeddings
Coming soon...

## Adding a new dataset
Instructions coming soon...

## Dependencies
The following python packages are used in seq2seq-chatbot:
(excluding packages that come with Anaconda)

- [TensorFlow](https://www.tensorflow.org/) 
  - **[pip install --upgrade tensorflow]** or **[pip install --upgrade tensorflow-gpu]**
- [jsonpickle](https://jsonpickle.github.io/)
  - **[pip install --upgrade jsonpickle]**
  
## Attribution
This implementation was inspired by:
- Kirill Eremenko & Hadelin de Ponteves [Deep NLP Udemy course](https://www.udemy.com/chatbot/)
- TensorFlow's [Neural Machine Translation (seq2seq) Tutorial](https://www.tensorflow.org/tutorials/seq2seq)
  - [TF NMT GitHub](https://github.com/tensorflow/nmt)
  
