"""
ChatSettings class
"""
import copy

class ChatSettings(object):
    """Contains settings for a chat session.
    """
    def __init__(self, inference_hparams):
        """
        Args:
            inference_hparams: the loaded InferenceHparams instance to use as default for this chat session
        """
        self.show_question_context = False
        self.show_all_beams = False
        self.inference_hparams = None
        
        self._default_inference_hparams = inference_hparams
        self.reset_to_defaults()
    
    def reset_to_defaults(self):
        """Reset all settings to defaults
        """
        self.show_question_context = False
        self.show_all_beams = False
        self.inference_hparams = copy.copy(self._default_inference_hparams)