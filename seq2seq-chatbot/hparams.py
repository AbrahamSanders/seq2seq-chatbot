"""
Hyperparameters class
"""

import jsonpickle
from vocabulary_importers.vocabulary_importer import VocabularyImportMode

class Hparams(object):
    """Container for model, training, and inference hyperparameters.

    Members:
        model_hparams: ModelHparams instance

        training_hparams: TrainingHparams instance

        inference_hparams: InferenceHparams instance
    """
    def __init__(self):
        """Initializes the Hparams instance.
        """
        self.model_hparams = ModelHparams()
        self.training_hparams = TrainingHparams()
        self.inference_hparams = InferenceHparams()
    
    @staticmethod 
    def load(filepath):
        """Loads the hyperparameters from a JSON file.

        Args:
            filepath: path of the JSON file.
        """
        with open(filepath, "r") as file:
            json = file.read()
        hparams = jsonpickle.decode(json)
        hparams.training_hparams.input_vocab_import_mode = VocabularyImportMode[hparams.training_hparams.input_vocab_import_mode]
        hparams.training_hparams.output_vocab_import_mode = VocabularyImportMode[hparams.training_hparams.output_vocab_import_mode]
        return hparams


class ModelHparams(object):
    """Hyperparameters which determine the architecture and complexity of the chatbot model.

    Members:
        rnn_cell_type: The architecture of RNN cell: "lstm" or "gru"
            LSTM: "Long-Short Term Memory"
            GRU: "Gated Recurrent Unit"
        
        rnn_size: the number of units (neurons) in each RNN cell. Applies to the encoder and decoder.
        
        use_bidirectional_encoder: True to use a bi-directional encoder.
            Bi-directional encoder: Two separate RNN cells (or stacks of cells) are used - 
                one receives the input sequence (question) in forward order, one receives the input sequence (question) in reverse order.
                When creating stacked RNN layers, each direction is stacked separately, with one stack for forward cells 
                and one stack for reverse cells.
            Uni-directional encoder: One RNN cell (or stack of cells) is used in the forward direction (traditional RNN)
        
        encoder_num_layers: the number of RNN cells to stack in the encoder.
            If use_bidirectional_encoder is set to true, this number is divided in half and applied to
            each direction. For example: 4 layers with bidrectional encoder means 2 forward & 2 backward cells.
        
        decoder_num_layers: the number of RNN cells to stack in the decoder.
            The encoder state can only be passed in to the decoder as its intial state if this value
            is the same as encoder_num_layers.
        
        encoder_embedding_size: the number of dimensions for each vector in the encoder embedding matrix.
            This matrix will be shaped (input_vocabulary.size(), encoder_embedding_size)
        
        decoder_embedding_size: the number of dimensions for each vector in the decoder embedding matrix.
            This matrix will be shaped (output_vocabulary.size(), decoder_embedding_size)

        encoder_embedding_trainable: True to allow gradient updates to be applied to the encoder embedding matrix.
            False to freeze the embedding matrix and only train the encoder & decoder RNNs, enabling greater training
            efficiency when loading pre-trained embeddings such as Word2Vec.

        decoder_embedding_trainable: True to allow gradient updates to be applied to the decoder embedding matrix.
            False to freeze the embedding matrix and only train the encoder & decoder RNNs, enabling greater training
            efficiency when loading pre-trained embeddings such as Word2Vec.
        
        share_embedding: True to reuse the same embedding matrix for the encoder and decoder.
            If the vocabulary is identical between input questions and output answers (as in a chatbot), then this should be True.
            If the vocabulary is different between input questions and output answers (as in a domain-specific Q&A system), then this should be False.
            If True - 
                1) input_vocabulary.size() & output_vocabulary.size() must have the same value
                2) encoder_embedding_size & decoder_embedding_size must have the same value
                3) encoder_embedding_trainable & decoder_embedding_trainable must have the same value
                4) If loading pre-trained embeddings, --encoderembeddingsdir & --decoderembeddingsdir args 
                    must be supplied with the same value (or --embeddingsdir can be used instead)
            If all of the above conditions are not met, an error is raised.
        
        attention_type: Type of attention mechanism to use. 
            ("bahdanau", "normed_bahdanau", "luong", "scaled_luong")
        
        beam_width: If mode is "infer", the number of beams to generate with the BeamSearchDecoder.
            Set to 0 for greedy / sampling decoding.
            This value is ignored if mode is "train".
            NOTE: this parameter should ideally be in InferenceHparams instead of ModelHparams, but is here for now
                because the graph of the model physically changes based on the beam width.
        
        enable_sampling: If True while beam_width = 0, the sampling decoder is used instead of the greedy decoder. 
        
        optimizer: Type of optimizer to use when training.
            ("sgd", "adam")
            NOTE: this parameter should ideally be in TrainingHparams instead of ModelHparams, but is here for now
                because the graph of the model physically changes based on which optimizer is used.

        max_gradient_norm: max value to clip the gradients if gradient clipping is enabled.
            Set to 0 to disable gradient clipping. Defaults to 5.
            This value is ignored if mode is "infer".
            NOTE: this parameter should ideally be in TrainingHparams instead of ModelHparams, but is here for now
                because the graph of the model physically changes based on whether or not gradient clipping is used.
        
        gpu_dynamic_memory_growth: Configures the TensorFlow session to only allocate GPU memory as needed,
            instead of the default behavior of trying to aggresively allocate as much memory as possible.
            Defaults to True.
    """
    def __init__(self):
        """Initializes the ModelHparams instance.
        """
        self.rnn_cell_type = "lstm"
        
        self.rnn_size = 256
        
        self.use_bidirectional_encoder = True
        
        self.encoder_num_layers = 2
        
        self.decoder_num_layers = 2
        
        self.encoder_embedding_size = 256
        
        self.decoder_embedding_size = 256

        self.encoder_embedding_trainable = True

        self.decoder_embedding_trainable = True
        
        self.share_embedding = True
        
        self.attention_type = "normed_bahdanau"
        
        self.beam_width = 10
        
        self.enable_sampling = False

        self.optimizer = "adam"
        
        self.max_gradient_norm = 5.
        
        self.gpu_dynamic_memory_growth = True
        
class TrainingHparams(object):
    """Hyperparameters used when training the chatbot model.
    
    Members:
        min_question_words: minimum length (in words) for a question.
            set this to a higher number if you wish to exclude shorter questions which
            can sometimes lead to higher training error.
        
        max_question_answer_words: maximum length (in words) for a question or answer.
            any questions or answers longer than this are truncated to fit. The higher this number, the more
            timesteps the encoder RNN will need to be unrolled.
        
        max_conversations: number of conversations to use from the cornell dataset. Specify -1 for no limit.
            pick a lower limit if training on the whole dataset is too slow (for lower-end GPUs)
        
        conv_history_length: number of conversation steps to prepend every question.
            For example, a length of 2 would output:
                "hello how are you ? <EOS> i am fine thank you <EOS> how is the new job?"
            where "how is the new job?" is the question and the rest is the prepended conversation history.
            the intent is to let the attention mechanism be able to pick up context clues from earlier in the
            conversation in order to determine the best way to respond.
            pick a lower limit if training is too slow or causes out of memory errors. The higher this number,
            the more timesteps the encoder RNN will need to be unrolled.

        normalize_words: True to preprocess the words in the training dataset by replacing word contractions 
            with their full forms (e.g. i'm -> i am) and then stripping out any remaining apostrophes.
        
        input_vocab_threshold: the minimum number of times a word must appear in the questions in order to be included
            in the vocabulary embedding. Any words that are not included in the vocabulary
            get replaced with an <OUT> token before training and inference.
            if model_params.share_embedding = True, this must equal output_vocab_threshold.

        output_vocab_threshold: the minimum number of times a word must appear in the answers in order to be included
            For more info see input_vocab_threshold.
            if model_params.share_embedding = True, this must equal input_vocab_threshold.

        input_vocab_import_normalized: True to normalize external word vocabularies and embeddings before import as input vocabulary.
            In this context normalization means convert all word tokens to lower case and then average the embedding vectors for any duplicate words.
            For example, "JOHN", "John", and "john" will be converted to "john" and it will take the mean of all three embedding vectors.

        output_vocab_import_normalized: True to normalize external word vocabularies and embeddings before import as output vocabulary.
            For more info see input_vocab_import_normalized.

        input_vocab_import_mode: Mode to govern how external vocabularies and embeddings are imported as input vocabulary.
            Ignored if no external vocabulary specified.
            See VocabularyImportMode.

        output_vocab_import_mode: Mode to govern how external vocabularies and embeddings are imported as output vocabulary.
            Ignored if no external vocabulary specified.
            See VocabularyImportMode.
            This should be set to 'Dataset' or 'ExternalIntersectDataset' for large vocabularies, since the size of the
                decoder output layer is the vocabulary size. For example, an external embedding may have a vocabulary size 
                of 1 million, but only 30k words appear in the dataset and having an output layer of 30k dimensions is 
                much more efficient than an output layer of 1m dimensions.
        
        validation_set_percent: the percentage of the training dataset to use as the validation set.
        
        random_train_val_split: 
            True to split the dataset randomly. 
            False to split the dataset sequentially 
                (validation samples are the last N samples, where N = samples * (val_percent / 100))
        
        validation_metric: the metric to use to measure the model during validation.
            "loss" - cross-entropy loss between predictions and targets
            "accuracy" (coming soon)
            "bleu" (coming soon)
        
        epochs: Number of epochs to train (1 epoch = all samples in dataset)
        
        early_stopping_epochs: stop early if no improvement in the validation metric
            after training for the given number of epochs in a row.
        
        batch_size: Training batch size
        
        learning_rate: learning rate used by SGD.
        
        learning_rate_decay: rate at which the learning rate drops.
            for each epoch, current_lr = starting_lr * (decay) ^ (epoch - 1)
        
        min_learning_rate: lowest value that the learning rate can go.
        
        dropout: probability that any neuron will be temporarily disabled during any training iteration.
            this is a regularization technique that helps the model learn more independent correlations in the data 
            and can reduce overfitting.
        
        checkpoint_on_training: Write a checkpoint after an epoch if the training loss improved.
        
        checkpoint_on_validation: Write a checkpoint after an epoch if the validation metric improved.
        
        log_summary: True to log training stats & graph for visualization in tensorboard.

        log_cleaned_dataset: True to save a copy of the cleaned dataset to disk for debugging purposes before training begins.

        log_training_data: True to save a copy of the training question-answer pairs as represented by their vocabularies to disk.
            this is useful to see how frequently words are replaced by <OUT> and also how dialog context is prepended to questions.

        stats_after_n_batches: Output training statistics (loss, time, etc.) after every N batches.

        backup_on_training_loss: List of training loss values upon which to backup the model
            Backups are full copies of the latest checkpoint files to another directory, also including vocab and hparam files.
    """
    def __init__(self):
        """Initializes the TrainingHparams instance.
        """        
        self.min_question_words = 1
        
        self.max_question_answer_words = 30
        
        self.max_conversations = -1
        
        self.conv_history_length = 6

        self.normalize_words = True
        
        self.input_vocab_threshold = 2

        self.output_vocab_threshold = 2

        self.input_vocab_import_normalized = True

        self.output_vocab_import_normalized = True

        self.input_vocab_import_mode = VocabularyImportMode.External

        self.output_vocab_import_mode = VocabularyImportMode.Dataset
        
        self.validation_set_percent = 0
        
        self.random_train_val_split = True
        
        self.validation_metric = "loss"
        
        self.epochs = 500
        
        self.early_stopping_epochs = 500
        
        self.batch_size = 128
        
        self.learning_rate = 2.0
        
        self.learning_rate_decay = 0.99
        
        self.min_learning_rate = 0.1
        
        self.dropout = 0.2
        
        self.checkpoint_on_training = True
        
        self.checkpoint_on_validation = True
        
        self.log_summary = True

        self.log_cleaned_dataset = True

        self.log_training_data = True

        self.stats_after_n_batches = 100

        self.backup_on_training_loss = []
        
class InferenceHparams(object):
    """Hyperparameters used when chatting with the chatbot model (a.k.a prediction or inference).

    Members:        
        beam_length_penalty_weight: higher values mean longer beams are scored better
            while lower (or negative) values mean shorter beams are scored better.
            Ignored if beam_width = 0
        
        sampling_temperature: This value sets the softmax temperature of the sampling decoder, if enabled.
        
        max_answer_words: Max length (in words) for an answer.
        
        conv_history_length: number of conversation steps to prepend every question.
            This can be different from the value used during training.

        normalize_words: True to preprocess the words in the input question by replacing word contractions 
            with their full forms (e.g. i'm -> i am) and then stripping out any remaining apostrophes.
        
        log_summary: True to log attention alignment images and inference graph for visualization in tensorboard.

        log_chat: True to log conversation history (chatlog) to a file.
    """
    def __init__(self):
        """Initializes the InferenceHparams instance.
        """        
        self.beam_length_penalty_weight = 1.25
        
        self.sampling_temperature = 0.5
        
        self.max_answer_words = 100
        
        self.conv_history_length = 6

        self.normalize_words = True
        
        self.log_summary = True

        self.log_chat = True

