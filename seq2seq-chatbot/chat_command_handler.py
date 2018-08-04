"""
Command handler for chat session
"""
import os

def append_to_chatlog(chatlog_filepath, question, answer):
    """Append a question and answer to the chat log.

    Args:
        chatlog_filepath: Path to the chat log file

        question: the question string entered by the user

        answer: the answer string returned by the chatbot
            If chat_settings.show_all_beams = True, answer is the array of all answer beams with one string per beam.
    """
    chatlog_dir = os.path.dirname(chatlog_filepath)
    if not os.path.isdir(chatlog_dir):
        os.makedirs(chatlog_dir)
    with open(chatlog_filepath, "a", encoding="utf-8") as file:
        file.write("You: {0}".format(question))
        file.write('\n')
        file.write("ChatBot: {0}".format(answer))
        file.write('\n\n')

def print_commands():
    """Print the list of available commands and their descriptions.
    """
    print()
    print()
    print("Commands:")
    print("-----------General-----------------")
    print("--help (Show this list of commands)                             --reset (Reset to default settings from hparams.json [*]);")                               
    print("--exit (Quit);")
    print()
    print("-----------Chat Options:-----------")
    print("--enableautopunct (Auto add punctuation to questions);          --disableautopunct (Enter punctuation exactly as typed);")
    print("--enablenormwords (Auto replace 'don't' with 'do not', etc.);   --disablenormwords (Enter words exactly as typed);")
    print("--showquestioncontext (Show conversation history as context);   --hidequestioncontext (Show questions only);")
    print("--showbeams (Output all predicted beams);                       --hidebeams (Output only the highest ranked beam);")
    print("--convhistlength=N (Set conversation history length to N);      --clearconvhist (Clear history and start a new conversation);")
    print()
    print("-----------Model Options:----------")
    print("--beamwidth=N (Set beam width to N. 0 disables beamsearch [*]); --beamlenpenalty=N (Set beam length penalty to N);")
    print("--enablesampling (Use sampling decoder if beamwidth=0 [*]);     --disableasampling (Use greedy decoder if beamwidth=0 [*]);")
    print("--samplingtemp=N (Set sampling temperature to N);               --maxanswerlen=N (Set max words in answer to N);") 
    print()
    print()
    print("[*] Causes model to reload")
    print()
    print()

def handle_command(input_str, model, chat_settings):
    """Given a user input string, determine if it is a command or a question and process if it is a command.

    Args: 
        input_str: the user input string

        model: the ChatbotModel instance
        
        chat_settings: the ChatSettings instance
    """
    reload_model = False
    terminate_chat = False
    is_command = True
    cmd_value = _get_command_value(input_str)
    #General Commands
    if input_str == '--help':
        print_commands()
    elif input_str == '--reset':
        chat_settings.reset_to_defaults()
        reload_model = True
        print ("[Reset to default settings.]")
    elif input_str == '--exit':
        terminate_chat = True
    #Chat Options
    elif input_str == '--enableautopunct':
        chat_settings.enable_auto_punctuation = True
        print ("[Auto-punctuation enabled.]")
    elif input_str == '--disableautopunct':
        chat_settings.enable_auto_punctuation = False
        print ("[Auto-punctuation disabled.]")
    elif input_str == '--enablenormwords':
        chat_settings.inference_hparams.normalize_words = True
        print ("[Word normalization enabled.]")
    elif input_str == '--disablenormwords':
        chat_settings.inference_hparams.normalize_words = False
        print ("[Word normalization disabled.]")
    elif input_str == '--showquestioncontext':
        chat_settings.show_question_context = True
        print ("[Show question context enabled.]")
    elif input_str == "--hidequestioncontext":
        chat_settings.show_question_context = False
        print ("[Show question context disabled.]")
    elif input_str == '--showbeams':
        chat_settings.show_all_beams = True
        print ("[Show all beams enabled.]")
    elif input_str == "--hidebeams":
        chat_settings.show_all_beams = False
        print ("[Show all beams disabled.]")
    elif input_str.startswith("--convhistlength"):
        if cmd_value is not None:
            chat_settings.inference_hparams.conv_history_length = int(cmd_value)
            model.trim_conversation_history(chat_settings.inference_hparams.conv_history_length)
        print ("[Conversation history length set to {0}.]".format(chat_settings.inference_hparams.conv_history_length))
    elif input_str == '--clearconvhist':
        model.trim_conversation_history(0)
        print ("[Conversation history cleared.]")
    #Model Options
    elif input_str.startswith("--beamwidth"):
        if cmd_value is not None:
            chat_settings.model_hparams.beam_width = int(cmd_value)
            reload_model = True
        print ("[Beam width set to {0}.]".format(chat_settings.model_hparams.beam_width))
    elif input_str.startswith("--beamlenpenalty"):
        if cmd_value is not None:
            chat_settings.inference_hparams.beam_length_penalty_weight = float(cmd_value)
        print ("[Beam length penalty weight set to {0}.]".format(chat_settings.inference_hparams.beam_length_penalty_weight))
    elif input_str == '--enablesampling':
        chat_settings.model_hparams.enable_sampling = True
        if chat_settings.model_hparams.beam_width == 0:
            reload_model = True
        print ("[Sampling decoder enabled (if beamwidth=0).]")
    elif input_str == '--disablesampling':
        chat_settings.model_hparams.enable_sampling = False
        if chat_settings.model_hparams.beam_width == 0:
            reload_model = True
        print ("[Sampling decoder disabled. Using greedy decoder (if beamwidth=0).]")
    elif input_str.startswith("--samplingtemp"):
        if cmd_value is not None:
            chat_settings.inference_hparams.sampling_temperature = float(cmd_value)
        print ("[Sampling temperature set to {0}.]".format(chat_settings.inference_hparams.sampling_temperature))
    elif input_str.startswith("--maxanswerlen"):
        if cmd_value is not None:
            chat_settings.inference_hparams.max_answer_words = int(cmd_value)
        print ("[Max words in answer set to {0}.]".format(chat_settings.inference_hparams.max_answer_words))
    #Not a command
    else:
        is_command = False

    return is_command, terminate_chat, reload_model

def _get_command_value(input_str):
    """Parses a command string and returns the value to the right of the '=' sign

    Args:
        input_str: the command string
    """
    idx = input_str.find("=")
    if idx > -1:
        return input_str[idx+1:].strip()
    else:
        return None