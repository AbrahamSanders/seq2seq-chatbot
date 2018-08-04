"""
ChatSettings class
"""
import copy

class ChatSettings(object):
    """Contains settings for a chat session.
    """
    def __init__(self, model_hparams, inference_hparams):
        """
        Args:
            inference_hparams: the loaded InferenceHparams instance to use as default for this chat session
        """
        self.show_question_context = False
        self.show_all_beams = False
        self.enable_auto_punctuation = True
        self.model_hparams = None
        self.inference_hparams = None
        
        self._default_model_hparams = model_hparams
        self._default_inference_hparams = inference_hparams
        self.reset_to_defaults()
    
    def reset_to_defaults(self):
        """Reset all settings to defaults
        """
        self.show_question_context = False
        self.show_all_beams = False
        self.enable_auto_punctuation = True
        self.model_hparams = copy.copy(self._default_model_hparams)
        self.inference_hparams = copy.copy(self._default_inference_hparams)