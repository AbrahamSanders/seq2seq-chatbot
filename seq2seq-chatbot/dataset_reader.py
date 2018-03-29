"""
Base class for dataset readers
"""
import abc
from os import path

from dataset import Dataset
from vocabulary import Vocabulary

class DatasetReader(object):
    """Base class for dataset readers
    """

    def __init__(self, dataset_name):
        """Initialize the DatasetReader.

        Args:
            dataset_name: Name of the dataset. Subclass must pass this in.
        """
        self.dataset_name = dataset_name

    @abc.abstractmethod
    def _get_dialog_lines_and_conversations(self, dataset_dir):
        """Subclass must implement this
        
        Read the raw dataset files and extract a dictionary of dialog lines and a list of conversations.
        A conversation is a list of dictionary keys for dialog lines that sequentially form a conversation.
        
        Args:
          dataset_dir: directory to load the raw dataset file(s) from
        """
        pass
    
    def read_dataset(self, dataset_dir, model_dir, training_hparams, share_vocab = True):
        """Read and return a chatbot dataset based on the specified dataset
        
        Args:
          dataset_dir: directory to load the raw dataset file(s) from

          model_dir: directory to save the vocabulary to

          training_hparams: training parameters which determine how the dataset will be read.
          See hparams.py for in-depth comments.

          share_vocab: True to generate a single vocabulary file from the question and answer words.
            False to generate separate input and output vocabulary files, from the question and answer words respectively.
            (If training_hparams.conv_history_length > 0, share_vocab must be True since previous answers will be appended to the questions.)
        """
        
        if not share_vocab and training_hparams.conv_history_length > 0:
            raise ValueError("If training_hparams.conv_history_length > 0, share_vocab must be True since previous answers will be appended to the questions.")
        if share_vocab and training_hparams.input_vocab_threshold != training_hparams.output_vocab_threshold:
            raise ValueError("Cannot share vocabulary when the input and output vocab thresholds are different.")

        #Get dialog line and conversation collections
        id2line, conversations_ids = self._get_dialog_lines_and_conversations(dataset_dir)
        
        #Clean dialog lines
        for line_id in id2line:
            id2line[line_id] = Vocabulary.clean_text(id2line[line_id], training_hparams.max_question_answer_words)
                
        # Getting separately the questions and the answers
        questions_for_count = []
        questions = []
        answers = []
        for conversation in conversations_ids[:training_hparams.max_conversations]:
            for i in range(len(conversation) - 1):
                conv_up_to_question = ''
                for j in range(max(0, i - training_hparams.conv_history_length), i):
                    conv_up_to_question += id2line[conversation[j]] + " {0} ".format(Vocabulary.EOS)
                question = id2line[conversation[i]]
                question_with_history = conv_up_to_question + question
                answer = id2line[conversation[i+1]]
                if training_hparams.min_question_words <= len(question_with_history.split()):
                    questions.append(conv_up_to_question + question)
                    questions_for_count.append(question)
                    answers.append(answer)

        # Create the vocabulary object & add the question & answer words
        if share_vocab:
            questions_and_answers = []
            for i in range(len(questions_for_count)):
                question = questions_for_count[i]
                answer = answers[i]
                if i == 0 or question != answers[i - 1]:
                    questions_and_answers.append(question)
                questions_and_answers.append(answer)
            
            input_vocabulary = self._create_and_save_vocab(questions_and_answers, training_hparams.input_vocab_threshold, model_dir, Vocabulary.SHARED_VOCAB_FILENAME)
            output_vocabulary = input_vocabulary
        else:
            input_vocabulary = self._create_and_save_vocab(questions_for_count, training_hparams.input_vocab_threshold, model_dir, Vocabulary.INPUT_VOCAB_FILENAME)
            output_vocabulary = self._create_and_save_vocab(answers, training_hparams.output_vocab_threshold, model_dir, Vocabulary.OUTPUT_VOCAB_FILENAME)
        
        # Adding the End Of String tokens to the end of every answer
        for i in range(len(answers)):
            answers[i] += " {0}".format(Vocabulary.EOS)
        
        #Create the Dataset object from the questions / answers lists and the vocab object.
        dataset = Dataset(questions, answers, input_vocabulary, output_vocabulary)
        
        return dataset

    def _create_and_save_vocab(self, word_sequences, vocab_threshold, model_dir, vocab_filename):
        """Create a Vocabulary instance from a list of word sequences, and save it to disk.

        Args:
            word_sequences: List of word sequences (sentence(s)) to use as basis for the vocabulary.

            vocab_threshold: Minimum number of times any word must appear within word_sequences 
                in order to be included in the vocabulary.
        """
        vocabulary = Vocabulary()
        for i in range(len(word_sequences)):
            word_seq = word_sequences[i]
            vocabulary.add_words(word_seq.split())
        vocabulary.compile(vocab_threshold)

        vocab_filepath = path.join(model_dir, vocab_filename)
        vocabulary.save(vocab_filepath)
        return vocabulary
