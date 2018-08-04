"""
ChatbotModel class
"""
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from tensorflow.contrib.tensorboard.plugins import projector
from os import path

from vocabulary import Vocabulary

class ChatbotModel(object):
    """Seq2Seq chatbot model class.

    This class encapsulates all interaction with TensorFlow. Users of this class need not be aware of the dependency on TF.
    This abstraction allows for future use of alternative DL libraries* by writing ChatbotModel implementations for them
    without the calling code needing to change.

    This class implementation was influenced by studying the TensorFlow Neural Machine Translation (NMT) tutorial:
        - https://www.tensorflow.org/tutorials/seq2seq
    These files were helpful in understanding the seq2seq implementation:
        - https://github.com/tensorflow/nmt/blob/master/nmt/model.py
        - https://github.com/tensorflow/nmt/blob/master/nmt/model_helper.py
        - https://github.com/tensorflow/nmt/blob/master/nmt/attention_model.py

    Although the code in this implementation is original, there are some similarities with the NMT tutorial in certain places.
    """
    
    def __init__(self,
                 mode,
                 model_hparams,
                 input_vocabulary,
                 output_vocabulary,
                 model_dir):
        """Create the Seq2Seq chatbot model.
        
        Args:
            mode: "train" or "infer"
            
            model_hparams: parameters which determine the architecture and complexity of the model.
                See hparams.py for in-depth comments.
            
            input_vocabulary: the input vocabulary. Each word in this vocabulary gets its own vector
                in the encoder embedding matrix.
                
            output_vocabulary: the output vocabulary. Each word in this vocabulary gets its own vector
                in the decoder embedding matrix.
                
            model_dir: the directory to output model summaries and load / save checkpoints.
        """
        self.mode = mode
        self.model_hparams = model_hparams
        self.input_vocabulary = input_vocabulary
        self.output_vocabulary = output_vocabulary
        self.model_dir = model_dir

        #Validate arguments
        if self.model_hparams.share_embedding and self.input_vocabulary.size() != self.output_vocabulary.size():
            raise ValueError("Cannot share embedding matrices when the input and output vocabulary sizes are different.")
        
        if self.model_hparams.share_embedding and self.model_hparams.encoder_embedding_size != self.model_hparams.decoder_embedding_size:
            raise ValueError("Cannot share embedding matrices when the encoder and decoder embedding sizes are different.")

        if self.input_vocabulary.external_embeddings is not None and self.input_vocabulary.external_embeddings.shape[1] != self.model_hparams.encoder_embedding_size:
            raise ValueError("Cannot use external embeddings with vector size {0} for the encoder when the hparams encoder embedding size is {1}".format(self.input_vocabulary.external_embeddings.shape[1],
                                                                                                                                                         self.model_hparams.encoder_embedding_size))
        if self.output_vocabulary.external_embeddings is not None and self.output_vocabulary.external_embeddings.shape[1] != self.model_hparams.decoder_embedding_size:
            raise ValueError("Cannot use external embeddings with vector size {0} for the decoder when the hparams decoder embedding size is {1}".format(self.output_vocabulary.external_embeddings.shape[1],
                                                                                                                                                         self.model_hparams.decoder_embedding_size))
                                                                                                                                            
        tf.contrib.learn.ModeKeys.validate(self.mode)
        
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN or self.model_hparams.beam_width is None:
            self.beam_width = 0
        else:
            self.beam_width = self.model_hparams.beam_width
        
        #Reset the default TF graph
        tf.reset_default_graph()
        
        #Define general model inputs
        self.inputs = tf.placeholder(tf.int32, [None, None], name = "inputs")
        self.input_sequence_length = tf.placeholder(tf.int32, [None], name = "input_sequence_length")
        
        #Build model
        initializer_feed_dict = {}
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            #Define training model inputs
            self.targets = tf.placeholder(tf.int32, [None, None], name = "targets")
            self.target_sequence_length = tf.placeholder(tf.int32, [None], name = "target_sequence_length")
            self.learning_rate = tf.placeholder(tf.float32, name= "learning_rate")
            self.keep_prob = tf.placeholder(tf.float32, name = "keep_prob")
            if self.input_vocabulary.external_embeddings is not None:
                self.input_external_embeddings = tf.placeholder(tf.float32, 
                                shape = self.input_vocabulary.external_embeddings.shape, 
                                name = "input_external_embeddings")
                initializer_feed_dict[self.input_external_embeddings] = self.input_vocabulary.external_embeddings
            if self.output_vocabulary.external_embeddings is not None and not self.model_hparams.share_embedding:
                self.output_external_embeddings = tf.placeholder(tf.float32, 
                                shape = self.output_vocabulary.external_embeddings.shape,
                                name = "output_external_embeddings")
                initializer_feed_dict[self.output_external_embeddings] = self.output_vocabulary.external_embeddings
            
            self.loss, self.training_step = self._build_model()
        
        elif self.mode == tf.contrib.learn.ModeKeys.INFER:
            #Define inference model inputs
            self.max_output_sequence_length = tf.placeholder(tf.int32, [], name="max_output_sequence_length")
            self.beam_length_penalty_weight = tf.placeholder(tf.float32, name = "beam_length_penalty_weight")
            self.sampling_temperature = tf.placeholder(tf.float32, name = "sampling_temperature")
            
            self.predictions, self.predictions_seq_lengths = self._build_model()
            self.conversation_history = []
        else:
            raise ValueError("Unsupported model mode. Choose 'train' or 'infer'.")

        # Get the final merged summary for writing to TensorBoard
        self.merged_summary = tf.summary.merge_all()
        
        # Defining the session, summary writer, and checkpoint saver
        self.session = self._create_session()
        
        self.summary_writer = tf.summary.FileWriter(self.model_dir, self.session.graph)

        self.session.run(tf.global_variables_initializer(), initializer_feed_dict)
        
        self.saver = tf.train.Saver()
    
    def load(self, filename):
        """Loads a trained model from a checkpoint
        
        Args:
            filename: Checkpoint filename, such as best_model_checkpoint.ckpt
                This file must exist within model_dir.
        """
        filepath = path.join(self.model_dir, filename)
        self.saver.restore(self.session, filepath)
    
    def save(self, filename):
        """Saves a checkpoint of the current model weights

        Args:
            filename: Checkpoint filename, such as best_model_checkpoint.ckpt.
                This file must exist within model_dir.
        """
        filepath = path.join(self.model_dir, filename)
        self.saver.save(self.session, filepath)

        config = projector.ProjectorConfig()
        if self.model_hparams.share_embedding:
            shared_embedding = config.embeddings.add()
            shared_embedding.tensor_name = "model/encoder/shared_embeddings_matrix"
            shared_embedding.metadata_path = Vocabulary.SHARED_VOCAB_FILENAME
        else:
            encoder_embedding = config.embeddings.add()
            encoder_embedding.tensor_name = "model/encoder/encoder_embeddings_matrix"
            encoder_embedding.metadata_path = Vocabulary.INPUT_VOCAB_FILENAME
            decoder_embedding = config.embeddings.add()
            decoder_embedding.tensor_name = "model/decoder/decoder_embeddings_matrix"
            decoder_embedding.metadata_path = Vocabulary.OUTPUT_VOCAB_FILENAME
        
        projector.visualize_embeddings(self.summary_writer, config)
    
    def train_batch(self, inputs, targets, input_sequence_length, target_sequence_length, learning_rate, dropout, global_step, log_summary=True):
        """Train the model on one batch, and return the training loss.

        Args:
            inputs: The input matrix of shape (batch_size, sequence_length)
                where each value in the sequences are words encoded as integer indexes of the input vocabulary.
            
            targets: The target matrix of shape (batch_size, sequence_length)
                where each value in the sequences are words encoded as integer indexes of the output vocabulary.

            input_sequence_length: A vector of sequence lengths of shape (batch_size)
                containing the lengths of every input sequence in the batch. This allows for dynamic sequence lengths.

            target_sequence_length: A vector of sequence lengths of shape (batch_size)
                containing the lengths of every target sequence in the batch. This allows for dynamic sequence lengths.

            learning_rate: The learning rate to use for the weight updates.

            dropout: The probability (0 <= p <= 1) that any neuron will be randomly disabled for this training step.
                This regularization technique can allow the model to learn more independent relationships in the input data
                and reduce overfitting. Too much dropout can make the model underfit. Typical values range between 0.2 - 0.5

            global_step: The index of this training step across all batches and all epochs. This allows TensorBoard to trend
                the training visually over time.

            log_summary: Flag indicating if the training summary should be logged (for visualization in TensorBoard).
        """
        
        if self.mode != tf.contrib.learn.ModeKeys.TRAIN:
            raise ValueError("train_batch can only be called when the model is initialized in train mode.")
        
        #Calculate the keep_probability (prob. a neuron will not be dropped) as 1 - dropout rate
        keep_probability = 1.0 - dropout

        #Train on the batch
        _, batch_training_loss, merged_summary = self.session.run([self.training_step, self.loss, self.merged_summary], 
                                                                    { self.inputs: inputs,
                                                                      self.targets: targets,
                                                                      self.input_sequence_length: input_sequence_length,
                                                                      self.target_sequence_length: target_sequence_length,
                                                                      self.learning_rate: learning_rate,
                                                                      self.keep_prob: keep_probability })

        #Write the training summary for this step if summary logging is enabled.
        if log_summary:                                                          
            self.summary_writer.add_summary(merged_summary, global_step)

        return batch_training_loss
    
    def validate_batch(self, inputs, targets, input_sequence_length, target_sequence_length, metric = "loss"):
        """Evaluate the metric on one batch and return.

        Args:
            inputs: The input matrix of shape (batch_size, sequence_length)
                where each value in the sequences are words encoded as integer indexes of the input vocabulary.
            
            targets: The target matrix of shape (batch_size, sequence_length)
                where each value in the sequences are words encoded as integer indexes of the output vocabulary.

            input_sequence_length: A vector of sequence lengths of shape (batch_size)
                containing the lengths of every input sequence in the batch. This allows for dynamic sequence lengths.

            target_sequence_length: A vector of sequence lengths of shape (batch_size)
                containing the lengths of every target sequence in the batch. This allows for dynamic sequence lengths.

            metric: The desired validation metric. Currently only "loss" is supported. This will eventually support
                "accuracy", "bleu", and other common validation metrics.
        """
        
        if self.mode != tf.contrib.learn.ModeKeys.TRAIN:
            raise ValueError("validate_batch can only be called when the model is initialized in train mode.")

        if metric == "loss":
            metric_op = self.loss
        else:
            raise ValueError("Unsupported validation metric: '{0}'".format(metric))
            
        metric_value = self.session.run(metric_op, { self.inputs: inputs,
                                                     self.targets: targets,
                                                     self.input_sequence_length: input_sequence_length,
                                                     self.target_sequence_length: target_sequence_length,
                                                     self.keep_prob: 1 })
        
        return metric_value
    
    def predict_batch(self, inputs, input_sequence_length, max_output_sequence_length, beam_length_penalty_weight, sampling_temperature, log_summary=True):
        """Predict a batch of output sequences given a batch of input sequences.
        
        Args:
            inputs: The input matrix of shape (batch_size, sequence_length)
                where each value in the sequences are words encoded as integer indexes of the input vocabulary.

            input_sequence_length: A vector of sequence lengths of shape (batch_size)
                containing the lengths of every input sequence in the batch. This allows for dynamic sequence lengths.

            max_output_sequence_length: The maximum number of timesteps the decoder can generate.
                If the decoder generates an EOS token sooner, it will end there. This maximum value just makes sure
                the decoder doesn't go on forever if no EOS is generated.

            beam_length_penalty_weight: When using beam search decoding, this penalty weight influences how
                beams are ranked. Large negative values rank very short beams first while large postive values rank very long beams first.
                A value of 0 will not influence the beam ranking. For a chatbot model, positive values between 0 and 2 can be beneficial
                to help the bot avoid short generic answers.

            sampling_temperature: When using sampling decoding, higher temperature values result in more random sampling
                while lower temperature values behave more like greedy decoding which takes the argmax of the output class distribution
                (softmax probability distribution over the output vocabulary). If this value is set to 0, sampling is disabled
                and greedy decoding is used.

            log_summary: Flag indicating if the inference summary should be logged (for visualization in TensorBoard).
        """
        
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            raise ValueError("predict_batch can only be called when the model is initialized in infer mode.")

        fetches = [{ "predictions": self.predictions, "predictions_seq_lengths": self.predictions_seq_lengths }]
        if self.merged_summary is not None:
            fetches.append(self.merged_summary)

        predicted_output_info = self.session.run(fetches, { self.inputs: inputs,
                                                            self.input_sequence_length: input_sequence_length,
                                                            self.max_output_sequence_length: max_output_sequence_length,
                                                            self.beam_length_penalty_weight: beam_length_penalty_weight,
                                                            self.sampling_temperature: sampling_temperature })

        #Write the training summary for this prediction if summary logging is enabled.
        if log_summary and len(predicted_output_info) == 2:
            merged_summary = predicted_output_info[1]                                                          
            self.summary_writer.add_summary(merged_summary)

        return predicted_output_info[0]
    
    def chat(self, question, chat_settings):
        """Chat with the chatbot model by predicting an answer to a question.
        'question' and 'answer' in this context are generic terms for the interactions in a dialog exchange
        and can be statements, remarks, queries, requests, or any other type of dialog speech.
        For example:
        Question: "How are you?"     Answer: "Fine."
        Question: "That's great."    Answer: "Yeah."

        Args:
            question: The input question for which the model should predict an answer.

            chat_settings: The ChatSettings instance containing the chat settings and inference hyperparameters

        Returns:
            q_with_hist: question with history if chat_settings.show_question_context = True otherwise None.

            answers: array of answer beams if chat_settings.show_all_beams = True otherwise the single selected answer.
            
        """
        #Process the question by cleaning it and converting it to an integer encoded vector
        if chat_settings.enable_auto_punctuation:
            question = Vocabulary.auto_punctuate(question)
        question = Vocabulary.clean_text(question, normalize_words = chat_settings.inference_hparams.normalize_words)
        question = self.input_vocabulary.words2ints(question)
        
        #Prepend the currently tracked steps of the conversation history separated by EOS tokens.
        #This allows for deeper dialog context to influence the answer prediction.
        question_with_history = []
        for i in range(len(self.conversation_history)):
            question_with_history += self.conversation_history[i] + [self.input_vocabulary.eos_int()]
        question_with_history += question
        
        #Get the answer prediction
        batch = np.zeros((1, len(question_with_history)))
        batch[0] = question_with_history
        max_output_sequence_length = chat_settings.inference_hparams.max_answer_words + 1 # + 1 since the EOS token is counted as a timestep
        predicted_answer_info = self.predict_batch(inputs = batch,
                                                   input_sequence_length = np.array([len(question_with_history)]),
                                                   max_output_sequence_length = max_output_sequence_length,
                                                   beam_length_penalty_weight = chat_settings.inference_hparams.beam_length_penalty_weight,
                                                   sampling_temperature = chat_settings.inference_hparams.sampling_temperature,
                                                   log_summary = chat_settings.inference_hparams.log_summary)
        
        #Read the answer prediction
        answer_beams = []
        if self.beam_width > 0:
            #For beam search decoding: if show_all_beams is enabeled then output all beams (sequences), otherwise take the first beam.
            #   The beams (in the "predictions" matrix) are ordered with the highest ranked beams first.
            beam_count = 1 if not chat_settings.show_all_beams else len(predicted_answer_info["predictions_seq_lengths"][0])
            for i in range(beam_count):
                predicted_answer_seq_length = predicted_answer_info["predictions_seq_lengths"][0][i] - 1 #-1 to exclude the EOS token
                predicted_answer = predicted_answer_info["predictions"][0][:predicted_answer_seq_length, i].tolist()
                answer_beams.append(predicted_answer)
        else:
            #For greedy / sampling decoding: only one beam (sequence) is returned, based on the argmax for greedy decoding
            #   or the sampling distribution for sampling decoding. Return this beam.
            beam_count = 1
            predicted_answer_seq_length = predicted_answer_info["predictions_seq_lengths"][0] - 1 #-1 to exclude the EOS token
            predicted_answer = predicted_answer_info["predictions"][0][:predicted_answer_seq_length].tolist()
            answer_beams.append(predicted_answer)
        
        #Add new conversation steps to the end of the history and trim from the beginning if it is longer than conv_history_length
        #Answers need to be converted from output_vocabulary ints to input_vocabulary ints (since they will be fed back in to the encoder)
        self.conversation_history.append(question)
        answer_for_history = self.output_vocabulary.ints2words(answer_beams[0], is_punct_discrete_word = True, capitalize_i = False)
        answer_for_history = self.input_vocabulary.words2ints(answer_for_history)
        self.conversation_history.append(answer_for_history)
        self.trim_conversation_history(chat_settings.inference_hparams.conv_history_length)
        
        #Convert the answer(s) to text and return
        answers = []
        for i in range(beam_count):
            answer = self.output_vocabulary.ints2words(answer_beams[i])
            answers.append(answer)
        
        q_with_hist = None if not chat_settings.show_question_context else self.input_vocabulary.ints2words(question_with_history)
        if chat_settings.show_all_beams:
            return q_with_hist, answers
        else:
            return q_with_hist, answers[0]
    
    def trim_conversation_history(self, length):
        """Trims the conversation history to the desired length by removing entries from the beginning of the array.
        This is the same conversation history prepended to each question to enable deep dialog context, so the shorter
        the length the less context the next question will have.

        Args:
            length: The desired length to trim the conversation history down to.
        """
        while len(self.conversation_history) > length:
            self.conversation_history.pop(0)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exception_type, exception_value, traceback):
        try:
            self.summary_writer.close()
        except:
            pass
        try:        
            self.session.close()
        except:
            pass
 
    def _build_model(self):
        """Create the seq2seq model graph.
        
        Since TensorFlow's default behavior is deferred execution, none of the tensor objects below actually have values until
        session.Run is called to train, validate, or predict a batch of inputs. 
        
        Eager execution was introduced in TF 1.5, but as of now this code does not use it.
        """
        with tf.variable_scope("model"):
            #Batch size for each batch is infered by looking at the first dimension of the input matrix.
            #While batch size is generally defined as a hyperparameter and does not change, in practice it can vary.
            #An example of this is if the number of samples in the training set is not evenly divisible by the batch size
            #in which case the last batch of each epoch will be smaller than the preset hyperparameter value.
            batch_size = tf.shape(self.inputs)[0]
            
            #encoder
            with tf.variable_scope("encoder"):
                #encoder_embeddings_matrix is a trainable matrix of values that contain the word embeddings for the input sequences.
                #   when a word is "embedded", it means that the input to the model is a dense N-dimensional vector that represents the word
                #   instead of a sparse one-hot encoded vector with the dimension of the word's index in the entire vocabulary set to 1.
                #   At training time, the dense embedding values that represent each word are updated in the direction of the loss gradient
                #   just like normal weights. Thus the model learns the contextual relationships between the words (the embedding) along with
                #   the objective function that depends on the words (the decoding).
                encoder_embeddings_matrix_name = "shared_embeddings_matrix" if self.model_hparams.share_embedding else "encoder_embeddings_matrix"
                encoder_embeddings_matrix_shape = [self.input_vocabulary.size(), self.model_hparams.encoder_embedding_size]
                encoder_embeddings_initial_value = (
                                self.input_external_embeddings if self.input_vocabulary.external_embeddings is not None 
                                else tf.random_uniform(encoder_embeddings_matrix_shape, 0, 1))
                encoder_embeddings_matrix = tf.Variable(encoder_embeddings_initial_value,
                                                        name = encoder_embeddings_matrix_name,
                                                        trainable = self.model_hparams.encoder_embedding_trainable,
                                                        expected_shape = encoder_embeddings_matrix_shape)
                    
                #As described above, the sequences of word vocabulary indexes in the inputs matrix are converted to sequences of 
                #N-dimensional dense vectors, by "looking them up" by index in the encoder_embeddings_matrix.
                encoder_embedded_input = tf.nn.embedding_lookup(encoder_embeddings_matrix, self.inputs)
                
                #Build the encoder RNN
                encoder_outputs, encoder_state = self._build_encoder(encoder_embedded_input)
            
            #Decoder
            with tf.variable_scope("decoder") as decoder_scope:
                #For description of word embeddings, see comments above on the encoder_embeddings_matrix.
                #If the share_embedding flag is set to True, the same matrix is used to embed words in the input and target sequences.
                #This is useful to avoid redundency when the same vocabulary is used for the inputs and targets
                # (why learn two ways to embed the same words?)
                if self.model_hparams.share_embedding:
                    decoder_embeddings_matrix = encoder_embeddings_matrix
                else:
                    decoder_embeddings_matrix_shape = [self.output_vocabulary.size(), self.model_hparams.decoder_embedding_size]
                    decoder_embeddings_initial_value = (
                                    self.output_external_embeddings if self.output_vocabulary.external_embeddings is not None 
                                    else tf.random_uniform(decoder_embeddings_matrix_shape, 0, 1))
                    decoder_embeddings_matrix = tf.Variable(decoder_embeddings_initial_value,
                                                            name = "decoder_embeddings_matrix",
                                                            trainable = self.model_hparams.decoder_embedding_trainable,
                                                            expected_shape = decoder_embeddings_matrix_shape)
                
                #Create the attentional decoder cell
                decoder_cell, decoder_initial_state = self._build_attention_decoder_cell(encoder_outputs, 
                                                                                        encoder_state,
                                                                                        batch_size)
                
                #Output (projection) layer
                weights = tf.truncated_normal_initializer(stddev=0.1)
                biases = tf.zeros_initializer()
                output_layer = layers_core.Dense(units = self.output_vocabulary.size(),
                                                kernel_initializer = weights,
                                                bias_initializer = biases,
                                                use_bias = True,
                                                name = "output_dense")
                
                #Build the decoder RNN using the attentional decoder cell and output layer
                if self.mode != tf.contrib.learn.ModeKeys.INFER:
                    #In train / validate mode, the training step and loss are returned. 
                    loss, training_step = self._build_training_decoder(batch_size,
                                                                       decoder_embeddings_matrix,
                                                                       decoder_cell, 
                                                                       decoder_initial_state, 
                                                                       decoder_scope, 
                                                                       output_layer)
                    return loss, training_step
                else:
                    #In inference mode, the predictions and prediction sequence lengths are returned.
                    #The sequence lengths can differ, but the predictions matrix will be one fixed size.
                    #The predictions_seq_lengths array can be used to properly read the sequences of variable lengths.
                    predictions, predictions_seq_lengths = self._build_inference_decoder(batch_size,
                                                                                         decoder_embeddings_matrix,
                                                                                         decoder_cell, 
                                                                                         decoder_initial_state, 
                                                                                         decoder_scope, 
                                                                                         output_layer)
                    return predictions, predictions_seq_lengths
                    
    def _build_encoder(self, encoder_embedded_input):
        """Create the encoder RNN

        Args:
            encoder_embedded_input: The embedded input sequences.
        """
        keep_prob = self.keep_prob if self.mode == tf.contrib.learn.ModeKeys.TRAIN else None
        if self.model_hparams.use_bidirectional_encoder:
            #Bi-directional encoding designates one or more RNN cells to read the sequence forward and one or more RNN cells to read
            #the sequence backward. The resulting states are concatenated before sending them on to the decoder.
            num_bi_layers = int(self.model_hparams.encoder_num_layers / 2)

            encoder_cell_forward = self._create_rnn_cell(self.model_hparams.rnn_size, num_bi_layers, keep_prob)
            encoder_cell_backward = self._create_rnn_cell(self.model_hparams.rnn_size, num_bi_layers, keep_prob)
            
            bi_encoder_outputs, bi_encoder_state = tf.nn.bidirectional_dynamic_rnn(
                                                                        cell_fw = encoder_cell_forward,
                                                                        cell_bw = encoder_cell_backward,
                                                                        sequence_length = self.input_sequence_length,
                                                                        inputs = encoder_embedded_input,
                                                                        dtype = tf.float32,
                                                                        swap_memory=True)
            
            #Manipulating encoder state to handle multi bidirectional layers
            encoder_outputs = tf.concat(bi_encoder_outputs, -1)
        
            if num_bi_layers == 1:
                encoder_state = bi_encoder_state
            else:
                # alternatively concat forward and backward states
                encoder_state = []
                for layer_id in range(num_bi_layers):
                    encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                    encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                encoder_state = tuple(encoder_state)
        
        else:
            #Uni-directional encoding uses all RNN cells to read the sequence forward.
            encoder_cell = self._create_rnn_cell(self.model_hparams.rnn_size, self.model_hparams.encoder_num_layers, keep_prob)
            
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                                                    cell = encoder_cell,
                                                    sequence_length = self.input_sequence_length,
                                                    inputs = encoder_embedded_input,
                                                    dtype = tf.float32,
                                                    swap_memory=True)
            
        return encoder_outputs, encoder_state
    
    def _build_attention_decoder_cell(self, encoder_outputs, encoder_state, batch_size):
        """Create the RNN cell to be used as the decoder and apply an attention mechanism.

        Args:
            encoder_outputs: a tensor containing the output of the encoder at each timestep of the input sequence.
                this is used as the input to the attention mechanism.
            
            encoder_state: a tensor containing the final encoder state for each encoder cell after reading the input sequence.
                if the encoder and decoder have the same structure, this becomes the decoder initial state.

            batch_size: the batch size tensor 
                (defined at the beginning of the model graph as the length of the first dimension of the input matrix)
        """
        #If beam search decoding - repeat the input sequence length, encoder output, encoder state, and batch size tensors
        #once for every beam.
        input_sequence_length = self.input_sequence_length
        if self.beam_width > 0:
            encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier = self.beam_width)
            input_sequence_length = tf.contrib.seq2seq.tile_batch(input_sequence_length, multiplier = self.beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier = self.beam_width)
            batch_size = batch_size * self.beam_width
        
        #Construct the attention mechanism
        if self.model_hparams.attention_type == "bahdanau" or self.model_hparams.attention_type == "normed_bahdanau":
            normalize = self.model_hparams.attention_type == "normed_bahdanau"
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units = self.model_hparams.rnn_size,
                                                                        memory = encoder_outputs,
                                                                        memory_sequence_length = input_sequence_length,
                                                                        normalize = normalize)
        elif self.model_hparams.attention_type == "luong" or self.model_hparams.attention_type == "scaled_luong":
            scale = self.model_hparams.attention_type == "scaled_luong"
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units = self.model_hparams.rnn_size,
                                                                    memory = encoder_outputs,
                                                                    memory_sequence_length = input_sequence_length,
                                                                    scale = scale)
        else:
            raise ValueError("Unsupported attention type. Use ('bahdanau' / 'normed_bahdanau') for Bahdanau attention or ('luong' / 'scaled_luong') for Luong attention.")
        
        #Create the decoder cell and wrap with the attention mechanism
        with tf.variable_scope("decoder_cell"):
            keep_prob = self.keep_prob if self.mode == tf.contrib.learn.ModeKeys.TRAIN else None
            decoder_cell = self._create_rnn_cell(self.model_hparams.rnn_size, self.model_hparams.decoder_num_layers, keep_prob)
        
            alignment_history = self.mode == tf.contrib.learn.ModeKeys.INFER and self.beam_width == 0
            output_attention = self.model_hparams.attention_type == "luong" or self.model_hparams.attention_type == "scaled_luong"
            attention_decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell = decoder_cell,
                                                                            attention_mechanism = attention_mechanism,
                                                                            attention_layer_size = self.model_hparams.rnn_size,
                                                                            alignment_history = alignment_history,
                                                                            output_attention = output_attention,
                                                                            name = "attention_decoder_cell")

        #If the encoder and decoder are the same structure, set the decoder initial state to the encoder final state.
        decoder_initial_state = attention_decoder_cell.zero_state(batch_size, tf.float32)
        if self.model_hparams.encoder_num_layers == self.model_hparams.decoder_num_layers:
            decoder_initial_state = decoder_initial_state.clone(cell_state = encoder_state)
    
        return attention_decoder_cell, decoder_initial_state
    
    def _build_training_decoder(self, batch_size, decoder_embeddings_matrix, decoder_cell, decoder_initial_state, decoder_scope, output_layer):
        """Build the decoder RNN for training mode.

        Currently this is implemented using the TensorFlow TrainingHelper, which uses the Teacher Forcing technique.
        "Teacher Forcing" means that at each output timestep, the next word of the target sequence is fed as input 
        to the decoder without regard to the output prediction at the previous timestep.

        Args:
            batch_size: the batch size tensor 
                (defined at the beginning of the model graph as the length of the first dimension of the input matrix)

            decoder_embeddings_matrix: The matrix containing the decoder embeddings

            decoder_cell: The RNN cell (or cells) used in the decoder.

            decoder_initial_state: The initial cell state of the decoder. This is the final encoder cell state if the encoder
                and decoder cells are structured the same. Otherwise it is a memory cell in zero state.
        """
        #Prepend each target sequence with the SOS token, as this is always the input of the first decoder timestep.
        preprocessed_targets = self._preprocess_targets(batch_size)
        #The sequences of word vocabulary indexes in the targets matrix are converted to sequences of 
        #N-dimensional dense vectors, by "looking them up" by index in the decoder_embeddings_matrix.
        decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)

        #Create the training decoder        
        helper = tf.contrib.seq2seq.TrainingHelper(inputs = decoder_embedded_input,
                                                   sequence_length = self.target_sequence_length)
    
        decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,
                                                  helper = helper,
                                                  initial_state = decoder_initial_state)
        
        #Get the decoder output
        decoder_output, _final_context_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder = decoder,
                                                                                                          swap_memory = True,
                                                                                                          scope = decoder_scope)
        #Pass the decoder output through the output dense layer which will become a softmax distribution of classes -
        #one class per word in the output vocabulary.
        logits = output_layer(decoder_output.rnn_output)
            
        #Calculate the softmax loss. Since the logits tensor is fixed size and the output sequences are variable length,
        #we need to "mask" the timesteps beyond the sequence length for each sequence. This multiplies the loss calculated
        #for those extra timesteps by 0, cancelling them out so they do not affect the final sequence loss.
        loss_mask = tf.sequence_mask(self.target_sequence_length, tf.shape(self.targets)[1], dtype=tf.float32)
        loss = tf.contrib.seq2seq.sequence_loss(logits = logits,
                                                targets = self.targets,
                                                weights = loss_mask)
        tf.summary.scalar("sequence_loss", loss)
        
        #Set up the optimizer
        if self.model_hparams.optimizer == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        elif self.model_hparams.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError("Unsupported optimizer. Use 'sgd' for GradientDescentOptimizer or 'adam' for AdamOptimizer.")

        tf.summary.scalar("learning_rate", self.learning_rate)
        #If max_gradient_norm is provided, enable gradient clipping.
        #This prevents an exploding gradient from derailing the training by too much.
        if self.model_hparams.max_gradient_norm > 0.0:
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, self.model_hparams.max_gradient_norm)
            tf.summary.scalar("gradient_norm", gradient_norm)
            tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients))
            training_step = optimizer.apply_gradients(zip(clipped_gradients, params))
        else:
            training_step = optimizer.minimize(loss = loss)
        
        return loss, training_step
    
    def _build_inference_decoder(self, batch_size, decoder_embeddings_matrix, decoder_cell, decoder_initial_state, decoder_scope, output_layer):
        """Build the decoder RNN for inference mode.

        In inference mode, the decoder takes the output of each timestep and feeds it in as the input to the next timestep
        and repeats this process until either an EOS token is generated or the maximum sequence length is reached.

        If beam_width > 0, beam search decoding is used.
            Beam search will sample from the top most likely words at each timestep and branch off (create a beam)
            on one or more of these words to explore a different version of the output sequence. For example:
                Question: How are you?
                Answer (beam 1): I am fine. <EOS>
                Answer (beam 2): I am doing well. <EOS>
                Answer (beam 3): I am doing alright considering the circumstances. <EOS>
                Answer (beam 4): Good and yourself? <EOS>
                Answer (beam 5): Good good. <EOS>
                Etc...
        
        If beam_width = 0, greedy or sampling decoding is used.

        Args:
            See _build_training_decoder
        """
        #Get the SOS and EOS tokens
        start_tokens = tf.fill([batch_size], self.output_vocabulary.sos_int())
        end_token = self.output_vocabulary.eos_int()
        
        #Build the beam search, greedy, or sampling decoder
        if self.beam_width > 0:
            decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell = decoder_cell,
                                                            embedding = decoder_embeddings_matrix,
                                                            start_tokens = start_tokens,
                                                            end_token = end_token,
                                                            initial_state = decoder_initial_state,
                                                            beam_width = self.beam_width,
                                                            output_layer = output_layer,
                                                            length_penalty_weight = self.beam_length_penalty_weight)
        else:
            if self.model_hparams.enable_sampling:
                helper = tf.contrib.seq2seq.SampleEmbeddingHelper(embedding = decoder_embeddings_matrix,
                                                                    start_tokens = start_tokens,
                                                                    end_token = end_token,
                                                                    softmax_temperature = self.sampling_temperature)
            else:
                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding = decoder_embeddings_matrix,
                                                                    start_tokens = start_tokens,
                                                                    end_token = end_token)
        
            decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell,
                                                        helper = helper,
                                                        initial_state = decoder_initial_state,
                                                        output_layer = output_layer)
        
        #Get the decoder output
        decoder_output, final_context_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder = decoder,
                                                                                                        maximum_iterations = self.max_output_sequence_length,
                                                                                                        swap_memory = True,
                                                                                                        scope = decoder_scope)
        
        #Return the predicted sequences along with an array of the sequence lengths for each predicted sequence in the batch
        if self.beam_width > 0:
            predictions = decoder_output.predicted_ids
            predictions_seq_lengths = final_context_state.lengths
        else:
            predictions = decoder_output.sample_id
            predictions_seq_lengths = final_sequence_lengths
            #Create attention alignment summary for visualization in TensorBoard
            self._create_attention_images_summary(final_context_state)
            
        return predictions, predictions_seq_lengths

    def _create_rnn_cell(self, rnn_size, num_layers, keep_prob):
        """Create a single RNN cell or stack of RNN cells (depending on num_layers)
        
        Args:
            rnn_size: number of units (neurons) in each RNN cell
            
            num_layers: number of stacked RNN cells to create

            keep_prob: probability of not being dropped out (1 - dropout), or None for inference mode
        """
        
        cells = []
        for _ in range(num_layers):
            if self.model_hparams.rnn_cell_type == "lstm":
                rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size)
            elif self.model_hparams.rnn_cell_type == "gru":
                rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_size)
            else:
                raise ValueError("Unsupported RNN cell type. Use 'lstm' for LSTM or 'gru' for GRU.")
            
            if keep_prob is not None:
                rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell, input_keep_prob=keep_prob)
                
            cells.append(rnn_cell)
            
        if len(cells) == 1:
            return cells[0]
        else:
            return tf.contrib.rnn.MultiRNNCell(cells = cells)
        
    def _preprocess_targets(self, batch_size):
        """Prepend the SOS token to all target sequences in the batch

        Args:
            batch_size: the batch size tensor 
                (defined at the beginning of the model graph as the length of the first dimension of the input matrix)
        """
        left_side = tf.fill([batch_size, 1], self.output_vocabulary.sos_int())
        right_side = tf.strided_slice(self.targets, [0,0], [batch_size, -1], [1,1])
        preprocessed_targets = tf.concat([left_side, right_side], 1)
        return preprocessed_targets
    
    def _create_session(self):
        """Initialize the TensorFlow session
        """
        if self.model_hparams.gpu_dynamic_memory_growth:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            session = tf.Session(config=config)
        else:
            session = tf.Session()
        
        return session

    def _create_attention_images_summary(self, final_context_state):
        """Create attention image and attention summary.
        
        TODO: this method was taken as is from the NMT tutorial and does not seem to work with the current model.
            figure this out and adjust as needed to the chatbot model.
        
        Args:
            final_context_state: final state of the decoder
        """
        attention_images = (final_context_state.alignment_history.stack())
        # Reshape to (batch, src_seq_len, tgt_seq_len,1)
        attention_images = tf.expand_dims(
            tf.transpose(attention_images, [1, 2, 0]), -1)
        # Scale to range [0, 255]
        attention_images *= 255
        tf.summary.image("attention_images", attention_images)