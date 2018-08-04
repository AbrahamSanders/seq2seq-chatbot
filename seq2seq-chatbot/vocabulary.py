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
    
    def __init__(self, external_embeddings = None):
        """Initializes the Vocabulary instance in an non-compiled state.
        Compile must be called before the Vocab instance can be used to integer encode/decode words.

        Args:
            external_embeddings: An optional 2d numpy array (matrix) containing external embedding vectors
        """
        self._word2count = {}
        self._words2int = {}
        self._ints2word = {}
        self._compiled = False
        self.external_embeddings = external_embeddings

    def load_word(self, word, word_int, count = 1):
        """Load a word and its integer encoding into the vocabulary instance. 

        Args:
            word: The word to load.

            word_int: The integer encoding of the word to load.

            count: (Optional) The number of times the word occurs in the source corpus.
        """
        self._validate_compile(False)

        self._word2count[word] = count
        self._words2int[word] = word_int
        self._ints2word[word_int] = word
        
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
    
    def compile(self, vocab_threshold = 1, loading = False):
        """Compile the internal lookup dictionaries that enable words to be integer encoded / decoded.

        Args:
            vocab_threshold: Minimum number of times any word must appear within word_sequences in order to be included in the vocabulary. 
                This is useful for filtering out rarely used words in order to reduce the size of the vocabulary 
                (which consequently reduces the size of the model's embedding matrices & reduces the dimensionality of the output softmax)
                This value is ignored if loading is True.

            loading: Indicates if the vocabulary is being loaded from disk, in which case the compilation is already done and this method
                only needs to set the flag to indicate as such.

            
        """
        self._validate_compile(False)
        
        if not loading:
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
    
    def word_exists(self, word):
        """Check if the given word exists in the vocabulary.

        Args:
            word: The word to check.
        """
        self._validate_compile(True)
        return word in self._words2int

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

    def ints2words(self, words_ints, is_punct_discrete_word = False, capitalize_i = True):
        """Decode a sequence of integers into a sequence of space delimited words

        Args:
            words_ints: The sequence of integers to decode

            is_punct_discrete_word: True to output a space before punctuation
                False to place punctuation immediately after the end of the preceeding word (normal usage).
        """
        words = ""
        for i in words_ints:
            word = self.int2word(i, capitalize_i)
            if is_punct_discrete_word or word not in ['.', '!', '?']:
                words += " "
            words += word
        words = words.strip()
        return words

    def int2word(self, word_int, capitalize_i = True):
        """Decode an integer into a word

        Args:
            words_int: The integer to decode
        """
        self._validate_compile(True)
        word = self._ints2word[word_int]
        if capitalize_i and word == 'i':
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
        with open(filepath, "w", encoding="utf-8") as file:
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
        
        with open(filepath, encoding="utf-8") as file:
            for index, line in enumerate(file):
                if index > 0: #Skip header line
                    word, count = line.split('\t')
                    word_int = index - 1
                    vocabulary.load_word(word, word_int, int(count))
        
        vocabulary.compile(loading = True)
        return vocabulary
    
    @staticmethod
    def clean_text(text, max_words = None, normalize_words = True):
        """Clean text to prepare for training and inference.
        
        Clean by removing unsupported special characters & extra whitespace,
        and by normalizing common word permutations (i.e. can't, cannot, can not)
        
        Args:
          text: the text to clean

          max_words: maximum number of words to output (assuming words are separated by spaces).
            any words beyond this limit are truncated.
            Defaults to None (unlimited number of words)

          normalize_words: True to replace word contractions with their full forms (e.g. i'm -> i am)
            and then strip out any remaining apostrophes.
        """
        text = text.lower()
        text = re.sub(r"'+", "'", text)
        if normalize_words:
            text = re.sub(r"i'm", "i am", text)
            text = re.sub(r"he's", "he is", text)
            text = re.sub(r"she's", "she is", text)
            text = re.sub(r"that's", "that is", text)
            text = re.sub(r"there's", "there is", text)
            text = re.sub(r"what's", "what is", text)
            text = re.sub(r"where's", "where is", text)
            text = re.sub(r"who's", "who is", text)
            text = re.sub(r"how's", "how is", text)
            text = re.sub(r"it's", "it is", text)
            text = re.sub(r"let's", "let us", text)
            text = re.sub(r"\'ll", " will", text)
            text = re.sub(r"\'ve", " have", text)
            text = re.sub(r"\'re", " are", text)
            text = re.sub(r"\'d", " would", text)
            text = re.sub(r"won't", "will not", text)
            text = re.sub(r"shan't", "shall not", text)
            text = re.sub(r"can't", "can not", text)
            text = re.sub(r"cannot", "can not", text)
            text = re.sub(r"n't", " not", text)
            text = re.sub(r"'", "", text)
        else:
            text = re.sub(r"(\W)'", r"\1", text)
            text = re.sub(r"'(\W)", r"\1", text)
        text = re.sub(r"[()\"#/@;:<>{}`+=~|$&*%\[\]_]", "", text)
        text = re.sub(r"[.]+", " . ", text)
        text = re.sub(r"[!]+", " ! ", text)
        text = re.sub(r"[?]+", " ? ", text)
        text = re.sub(r"[,-]+", " ", text)
        text = re.sub(r"[\t]+", " ", text)
        text = re.sub(r" +", " ", text)
        text = text.strip()
        
        #Truncate words beyond the limit, if provided. Remove partial sentences from the end if punctuation exists within the limit.
        if max_words is not None:
            text_parts = text.split()
            if len(text_parts) > max_words:
                truncated_text_parts = text_parts[:max_words]
                while len(truncated_text_parts) > 0 and not re.match("[.!?]", truncated_text_parts[-1]):
                    truncated_text_parts.pop(-1)
                if len(truncated_text_parts) == 0:
                    truncated_text_parts = text_parts[:max_words]
                text = " ".join(truncated_text_parts)
                
        return text

    @staticmethod
    def auto_punctuate(text):
        """Automatically apply punctuation to text that does not end with any punctuation marks.

        Args:
            text: the text to apply punctuation to.
        """
        text = text.strip()
        if not (text.endswith(".") or text.endswith("?") or text.endswith("!") or text.startswith("--")):
            tmp = re.sub(r"'", "", text.lower())
            if (tmp.startswith("who") or tmp.startswith("what") or tmp.startswith("when") or 
                    tmp.startswith("where") or tmp.startswith("why") or tmp.startswith("how") or
                    tmp.endswith("who") or tmp.endswith("what") or tmp.endswith("when") or 
                    tmp.endswith("where") or tmp.endswith("why") or tmp.endswith("how") or
                    tmp.startswith("are") or tmp.startswith("will") or tmp.startswith("wont") or tmp.startswith("can")):
                text = "{}?".format(text)
            else:
                text = "{}.".format(text)
        return text