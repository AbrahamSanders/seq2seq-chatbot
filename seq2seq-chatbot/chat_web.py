"""
Script for serving a trained chatbot model over http
"""
import datetime
import click
from os import path
from flask import Flask, request, send_from_directory
from flask_cors import CORS
from flask_restful import Resource, Api

import general_utils
import chat_command_handler
from chat_settings import ChatSettings
from chatbot_model import ChatbotModel
from vocabulary import Vocabulary

app = Flask(__name__)
CORS(app)

@app.cli.command()
@click.argument("checkpointfile")
@click.option("-p", "--port", type=int)
def serve_chat(checkpointfile, port):

    api = Api(app)

    #Read the hyperparameters and configure paths
    model_dir, hparams, checkpoint = general_utils.initialize_session_server(checkpointfile)

    #Load the vocabulary
    print()
    print ("Loading vocabulary...")
    if hparams.model_hparams.share_embedding:
        shared_vocab_filepath = path.join(model_dir, Vocabulary.SHARED_VOCAB_FILENAME)
        input_vocabulary = Vocabulary.load(shared_vocab_filepath)
        output_vocabulary = input_vocabulary
    else:
        input_vocab_filepath = path.join(model_dir, Vocabulary.INPUT_VOCAB_FILENAME)
        input_vocabulary = Vocabulary.load(input_vocab_filepath)
        output_vocab_filepath = path.join(model_dir, Vocabulary.OUTPUT_VOCAB_FILENAME)
        output_vocabulary = Vocabulary.load(output_vocab_filepath)

    #Create the model
    print ("Initializing model...")
    print()
    with ChatbotModel(mode = "infer",
                      model_hparams = hparams.model_hparams,
                      input_vocabulary = input_vocabulary,
                      output_vocabulary = output_vocabulary,
                      model_dir = model_dir) as model:

        #Load the weights
        print()
        print ("Loading model weights...")
        model.load(checkpoint)

        # Setting up the chat
        chatlog_filepath = path.join(model_dir, "chat_logs", "web_chatlog_{0}.txt".format(datetime.datetime.now().strftime("%Y%m%d_%H%M%S")))
        chat_settings = ChatSettings(hparams.model_hparams, hparams.inference_hparams)
        chat_command_handler.print_commands()
        
        class Answer(Resource):
            def get(self, question):
                is_command, terminate_chat, _ = chat_command_handler.handle_command(question, model, chat_settings)
                if terminate_chat:
                    answer = "[Can't terminate from http request]"
                elif is_command:
                    answer = "[Command processed]"
                else:
                    #If it is not a command (it is a question), pass it on to the chatbot model to get the answer
                    _, answer = model.chat(question, chat_settings)
                    
                    if chat_settings.inference_hparams.log_chat:
                        chat_command_handler.append_to_chatlog(chatlog_filepath, question, answer)

                return answer

        class UI(Resource):
            def get(self):
                return send_from_directory(".", "chat_ui.html")

        api.add_resource(Answer, "/chat/<string:question>")
        api.add_resource(UI, "/chat_ui/")
        app.run(debug=False, port=port)