"""
Vocabulary class
"""
import re

class Vocabulary(object):
    """Class representing a chatbot vocabulary.

    The Vocabulary class is responsible for encoding words into integers and decoding integers into words.
    The number of times each word occurs in the source corpus is also tracked for visualization purposes.

    Special tokens that exist in every vocabulary instance:
        - PAD ("<PAD>"): The token used for extra sequence timesteps in a batch
        - SOS ("<SOS>"): Start Of Sequence token is used as the input of the first decoder timestep
        - EOS ("<EOS>"): End Of Sequence token is used to signal that the decoder should stop generating a sequence.
                         It is also used to separate conversation history (context) questions prepended to the current input question.
        - OUT ("<OUT>"): If a word does not exist in the vocabulary, it is substituted with this token.
    """
    
    SHARED_VOCAB_FILENAME = "shared_vocab.tsv"
    INPUT_VOCAB_FILENAME = "input_vocab.tsv"
    OUTPUT_VOCAB_FILENAME = "output_vocab.tsv"

    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    OUT = "<OUT>"
    special_tokens = [PAD, SOS, EOS, OUT]
    
    def __init__(self):
        """Initializes the Vocabulary instance in an non-compiled state.
        Compile must be called before the Vocab instance can be used to integer encode/decode words.
        """
        self._word2count = {}
        self._words2int = {}
        self._ints2word = {}
        self._compiled = False

    def add_word(self, word, count = 1):
        """Add a word to the vocabulary instance. 
        Optionally specify the count of its occurrence in the source corpus.

        Args:
            word: The word to add.

            count: The number of times the word occurs in the source corpus.
        """
        self.add_words([word])
        if (count > 1):
            self._word2count[word] += (count - 1)  

    def add_words(self, words):
        """Add a sequence of words to the vocabulary instance.
        If a word occurs more than once, its count will be incremented accordingly.

        Args:
            words: The sequence of words to add.
        """
        self._validate_compile(False)

        for i in range(len(words)):
            word = words[i]
            if word in self._word2count:
                self._word2count[word] += 1
            else:
                self._word2count[word] = 1
    
    def compile(self, vocab_threshold = 1):
        """Compile the internal lookup dictionaries that enable words to be integer encoded / decoded.

        Args:
            vocab_threshold: Minimum number of times any word must appear within word_sequences in order to be included in the vocabulary. 
                This is useful for filtering out rarely used words in order to reduce the size of the vocabulary 
                (which consequently reduces the size of the model's embedding matrices & reduces the dimensionality of the output softmax)
        """
        self._validate_compile(False)
        
        #Add the special tokens to the lookup dictionaries
        for i, special_token in enumerate(Vocabulary.special_tokens):
            self._words2int[special_token] = i
            self._ints2word[i] = special_token

        #Add the words in _word2count to the lookup dictionaries if their count meets the threshold.
        #Any words that don't meet the threshold are removed.
        word_int = len(self._words2int)
        for word, count in sorted(self._word2count.items()):
            if count >= vocab_threshold:
                self._words2int[word] = word_int
                self._ints2word[word_int] = word
                word_int += 1
            else:
                del self._word2count[word]
        
        #Add the special tokens to _word2count so they have count values for saving to disk
        self.add_words(Vocabulary.special_tokens)

        #The Vocabulary instance may now be used for integer encoding / decoding
        self._compiled = True



    def size(self):
        """The size (number of words) of the Vocabulary
        """
        self._validate_compile(True)
        return len(self._word2count)
    
    def words2ints(self, words):
        """Encode a sequence of space delimited words into a sequence of integers

        Args:
            words: The sequence of space delimited words to encode
        """
        return [self.word2int(w) for w in words.split()]
    
    def word2int(self, word):
        """Encode a word into an integer

        Args:
            word: The word to encode
        """
        self._validate_compile(True)
        return self._words2int[word] if word in self._words2int else self.out_int()

    def ints2words(self, words_ints):
        """Decode a sequence of integers into a sequence of space delimited words

        Args:
            words_ints: The sequence of integers to decode
        """
        words = ""
        for i in words_ints:
            word = self.int2word(i)
            if word not in ['.', '!', '?']:
                words += " "
            words += word
        words = words.strip()
        return words

    def int2word(self, word_int):
        """Decode an integer into a word

        Args:
            words_int: The integer to decode
        """
        self._validate_compile(True)
        word = self._ints2word[word_int]       
        if word == 'i':
            word = 'I'
        return word
    
    def pad_int(self):
        """Get the integer encoding of the PAD token
        """
        return self.word2int(Vocabulary.PAD)

    def sos_int(self):
        """Get the integer encoding of the SOS token
        """
        return self.word2int(Vocabulary.SOS)
    
    def eos_int(self):
        """Get the integer encoding of the EOS token
        """
        return self.word2int(Vocabulary.EOS)

    def out_int(self):
        """Get the integer encoding of the OUT token
        """
        return self.word2int(Vocabulary.OUT)
    
    def save(self, filepath):
        """Saves the vocabulary to disk.

        Args:
            filepath: The path of the file to save to
        """
        total_words = self.size()
        with open(filepath, "w") as file:
            file.write('\t'.join(["word", "count"]))
            file.write('\n')
            for i in range(total_words):
                word = self._ints2word[i]
                count = self._word2count[word]
                file.write('\t'.join([word, str(count)]))
                if i < total_words - 1:
                    file.write('\n')

    def _validate_compile(self, expected_status):
        """Validate that the vocabulary is compiled or not based on the needs of the attempted operation

        Args:
            expected_status: The compilation status expected by the attempted operation
        """
        if self._compiled and not expected_status:
            raise ValueError("This vocabulary instance has already been compiled.")
        if not self._compiled and expected_status:
            raise ValueError("This vocabulary instance has not been compiled yet.")
    
    @staticmethod        
    def load(filepath):
        """Loads the vocabulary from disk.

        Args:
            filepath: The path of the file to load from
        """
        vocabulary = Vocabulary()
        #Skip header line + special token lines
        line_num_start = 1 + len(Vocabulary.special_tokens)

        with open(filepath, "r") as file:
            for line_num, line in enumerate(file):
                if line_num >= line_num_start: 
                    word, count = line.split('\t')
                    vocabulary.add_word(word, int(count))
        
        vocabulary.compile()
        return vocabulary
    
    @staticmethod
    def clean_text(text, max_words = None):
        """Clean text to prepare for training and inference.
        
        Clean by removing unsupported special characters & extra whitespace,
        and by normalizing common word permutations (i.e. can't, cannot, can not)
        
        Args:
          text: the text to clean

          max_words: maximum number of words to output (assuming words are separated by spaces).
            any words beyond this limit are truncated.
            Defaults to None (unlimited number of words)
        """
        text = text.lower()
        text = re.sub(r"'+", "'", text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"how's", "how is", text)
        text = re.sub(r"it's", "it is", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "can not", text)
        text = re.sub(r"cannot", "can not", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"[()\"#/@;:<>{}`'+=~|$&*%\[\]_]", "", text)
        text = re.sub(r"[.]+", " . ", text)
        text = re.sub(r"[!]+", " ! ", text)
        text = re.sub(r"[?]+", " ? ", text)
        text = re.sub(r"[,-]+", " ", text)
        text = re.sub(r"[\t]+", " ", text)
        text = re.sub(r" +", " ", text)
        text = text.strip()
        
        #Truncate words beyond the limit, if provided.
        if max_words is not None:
            text_parts = text.split()
            if len(text_parts) > max_words:
                text = " ".join(text_parts[:max_words])
                
        return text