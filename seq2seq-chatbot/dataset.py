"""
Dataset class
"""
import math
import random
import numpy as np
from os import path

class Dataset(object):
    """Class representing a chatbot dataset with questions, answers, and vocabulary.
    """
    
    def __init__(self, questions, answers, input_vocabulary, output_vocabulary):
        """Initializes a Dataset instance with a list of questions, answers, and input/output vocabularies.

        Args:
            questions: Can be a list of questions as space delimited sentence(s) of words
                or a list of lists of integer encoded words

            answers: Can be a list of answers as space delimited sentence(s) of words
                or a list of lists of integer encoded words

            input_vocabulary: The Vocabulary instance to use for encoding questions

            output_vocabulary: The Vocabulary instance to use for encoding answers
        """
        if len(questions) != len (answers):
            raise RuntimeError("questions and answers lists must be the same length, as they are lists of input-output pairs.")

        self.input_vocabulary = input_vocabulary
        self.output_vocabulary = output_vocabulary
        #If the questions and answers are already integer encoded, accept them as is.
        #Otherwise use the Vocabulary instances to encode the question and answer sequences.
        if len(questions) > 0 and isinstance(questions[0], str):
            self.questions_into_int = [self.input_vocabulary.words2ints(q) for q in questions]
            self.answers_into_int = [self.output_vocabulary.words2ints(a) for a in answers]
        else:
            self.questions_into_int = questions
            self.answers_into_int = answers
    
    def size(self):
        """ The size (number of samples) of the Dataset.
        """
        return len(self.questions_into_int)
    
    def train_val_split(self, val_percent = 20, random_split = True, move_samples = True):
        """Splits the dataset into training and validation sets.
        
        Args:
            val_percent: the percentage of the dataset to use as validation data.
            
            random_split: True to split the dataset randomly. 
                False to split the dataset sequentially (validation samples are the last N samples, where N = samples * (val_percent / 100))
            
            move_samples: True to physically move the samples into the returned training and validation dataset objects (saves memory).
                False to copy the samples into the returned training and validation dataset objects, and preserve this dataset instance.
        """
        
        if move_samples:
            questions = self.questions_into_int
            answers = self.answers_into_int
        else:
            questions = self.questions_into_int[:]
            answers = self.answers_into_int[:]
        
        num_validation_samples = int(len(questions) * (val_percent / 100))
        num_training_samples = len(questions) - num_validation_samples
        
        training_questions = []
        training_answers = []
        validation_questions = []
        validation_answers = []
        if random_split:
            for _ in range(num_validation_samples):
                random_index = random.randint(0, len(questions) - 1)
                validation_questions.append(questions.pop(random_index))
                validation_answers.append(answers.pop(random_index))
            
            for _ in range(num_training_samples):
                training_questions.append(questions.pop(0))
                training_answers.append(answers.pop(0))
        else:
            for _ in range(num_training_samples):
                training_questions.append(questions.pop(0))
                training_answers.append(answers.pop(0))
            
            for _ in range(num_validation_samples):
                validation_questions.append(questions.pop(0))
                validation_answers.append(answers.pop(0))
        
        training_dataset = Dataset(training_questions, training_answers, self.input_vocabulary, self.output_vocabulary)
        validation_dataset = Dataset(validation_questions, validation_answers, self.input_vocabulary, self.output_vocabulary)
        
        return training_dataset, validation_dataset
            
    def sort(self):
        """Sorts the dataset by the lengths of the questions. This can speed up training by reducing the
        amount of padding the input sequences need.
        """
        if self.size() > 0:
            self.questions_into_int, self.answers_into_int = zip(*sorted(zip(self.questions_into_int, self.answers_into_int), 
                                                                         key = lambda qa_pair: len(qa_pair[0])))

    def save(self, filepath):
        """Saves the dataset questions & answers exactly as represented by input_vocabulary and output_vocabulary.
        """
        filename, ext = path.splitext(filepath)
        questions_filepath = "{0}_questions{1}".format(filename, ext)
        answers_filepath = "{0}_answers{1}".format(filename, ext)

        with open(questions_filepath, mode="w", encoding="utf-8") as file:
            for question_into_int in self.questions_into_int:
                question = self.input_vocabulary.ints2words(question_into_int, is_punct_discrete_word = True, capitalize_i = False)
                file.write(question)
                file.write('\n')

        with open(answers_filepath, mode="w", encoding="utf-8") as file:
            for answer_into_int in self.answers_into_int:
                answer = self.output_vocabulary.ints2words(answer_into_int, is_punct_discrete_word = True, capitalize_i = False)
                file.write(answer)
                file.write('\n')


    
    def batches(self, batch_size):
        """Provide the dataset as an enumerable collection of batches of size batch_size.
        Each batch will be a matrix of a fixed shape (batch_size, max_seq_length_in_batch).
        Sequences that are shorter than the largest one are padded at the end with the PAD token.
        Padding is largely just used as a placeholder since the dyamic encoder and decoder RNNs 
        will never see the padded timesteps.

        Args:
            batch_size: size of each batch.
                If the total number of samples is not evenly divisible by batch_size, the last batch will contain the remainder
                which will be less than batch_size.

        Returns:
            padded_questions_in_batch: A list of padded, integer-encoded question sequences.

            padded_answers_in_batch: A list of padded, integer-encoded answer sequences.

            seqlen_questions_in_batch: A list of actual sequence lengths for each question in the batch.

            seqlen_answers_in_batch: A list of actual sequence lengths for each answer in the batch.
            
        """
        for batch_index in range(0, math.ceil(len(self.questions_into_int) / batch_size)):
                start_index = batch_index * batch_size
                questions_in_batch = self.questions_into_int[start_index : start_index + batch_size]
                answers_in_batch = self.answers_into_int[start_index : start_index + batch_size]
                
                seqlen_questions_in_batch = np.array([len(q) for q in questions_in_batch])
                seqlen_answers_in_batch = np.array([len(a) for a in answers_in_batch])
                
                padded_questions_in_batch = np.array(self._apply_padding(questions_in_batch, self.input_vocabulary))
                padded_answers_in_batch = np.array(self._apply_padding(answers_in_batch, self.output_vocabulary))
                
                yield padded_questions_in_batch, padded_answers_in_batch, seqlen_questions_in_batch, seqlen_answers_in_batch
    
    
    def _apply_padding(self, batch_of_sequences, vocabulary):
        """Padding the sequences with the <PAD> token to ensure all sequences in the batch are the same physical size.
        
        Input and target sequences can be any length, but each batch that is fed into the model
        must be defined as a fixed size matrix of shape (batch_size, max_seq_length_in_batch).
        
        Padding allows for this dynamic sequence length within a fixed matrix. 
        However, the actual padded sequence timesteps are never seen by the encoder or decoder nor are they 
        counted toward the softmax loss. This is possible since we provide the actual sequence lengths to the model 
        as a separate vector of shape (batch_size), where the RNNs are instructed to only unroll for the number of 
        timesteps that have real (unpadded) values. The sequence loss also accepts a masking weight matrix where 
        we can specify that loss values for padded timesteps should be ignored.

        Args:
            batch_of_sequences: list of integer-encoded sequences to pad

            vocabulary: Vocabulary instance to use to look up the integer encoding of the PAD token
        """
        max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
        return [sequence + ([vocabulary.pad_int()] * (max_sequence_length - len(sequence))) for sequence in batch_of_sequences]