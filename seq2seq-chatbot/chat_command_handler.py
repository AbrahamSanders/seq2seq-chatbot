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
    with open(chatlog_filepath, "a") as file:
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
    print("--showquestioncontext (Show questions with history as context); --hidequestioncontext (Show questions only);")
    print("--showbeams (Output all predicted beams);                       --hidebeams (Output only the highest ranked beam);")
    print("--beamlenpenalty=N (Set beam length penalty weight to N);       --samplingtemp=N (Set sampling temperature to N);")
    print("--maxanswerlen=N (Set max words in answer to N);                --convhistlength=N (Set the conversation history length to N);")
    print("--clearconvhist (Clear the conversation history);               --reset (Reset to default settings from hparams.json);")
    print("--help (Show this list of commands)                             --exit (Quit);")
    print()
    print()

def handle_command(input_str, model, chat_settings):
    """Given a user input string, determine if it is a command or a question and process if it is a command.

    Args: 
        input_str: the user input string

        model: the ChatbotModel instance
        
        chat_settings: the ChatSettings instance
    """
    terminate_chat = False
    is_command = True
    cmd_value = _get_command_value(input_str)
    if input_str == '--showquestioncontext':
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
    elif input_str.startswith("--beamlenpenalty"):
        if cmd_value is not None:
            chat_settings.inference_hparams.beam_length_penalty_weight = float(cmd_value)
        print ("[Beam length penalty weight set to {0}.]".format(chat_settings.inference_hparams.beam_length_penalty_weight))
    elif input_str.startswith("--samplingtemp"):
        if cmd_value is not None:
            chat_settings.inference_hparams.sampling_temperature = float(cmd_value)
        print ("[Sampling temperature set to {0}.]".format(chat_settings.inference_hparams.sampling_temperature))
    elif input_str.startswith("--maxanswerlen"):
        if cmd_value is not None:
            chat_settings.inference_hparams.max_answer_words = int(cmd_value)
        print ("[Max words in answer set to {0}.]".format(chat_settings.inference_hparams.max_answer_words))
    elif input_str.startswith("--convhistlength"):
        if cmd_value is not None:
            chat_settings.inference_hparams.conv_history_length = int(cmd_value)
            model.trim_conversation_history(chat_settings.inference_hparams.conv_history_length)
        print ("[Conversation history length set to {0}.]".format(chat_settings.inference_hparams.conv_history_length))
    elif input_str == '--clearconvhist':
        model.trim_conversation_history(0)
        print ("[Conversation history cleared.]")
    elif input_str == '--reset':
        chat_settings.reset_to_defaults()
        print ("[Reset to default settings.]")
    elif input_str == '--help':
        print_commands()
    elif input_str == '--exit':
        terminate_chat = True
    else:
        is_command = False

    return is_command, terminate_chat

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