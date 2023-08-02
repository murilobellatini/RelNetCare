import os
import time
import json
import random
import openai
from ast import literal_eval
from flask import Flask, render_template, request

from src.processing.neo4j_operations import DialogueGraphPersister
from src.paths import LOCAL_RAW_DATA_PATH


class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}

class DialogueLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save_message(self, message: Message, turn):
        message_data = {
            "turn": turn,
            "role": message.role,
            "content": message.content,
            "timestamp": time.time(),  # Timestamp in seconds since the Epoch
        }
        file_name = f"{str(message_data['turn']).zfill(3)}_{message.role}_{int(message_data['timestamp'])}.json"
        file_path = os.path.join(self.output_dir, file_name)
        with open(file_path, 'w') as file:
            json.dump(message_data, file, indent=2)

    def load_chat_history(self):
        files = sorted(os.listdir(self.output_dir)) 
        history = []
        for file in files:
            if file.endswith('.json'):  # To ensure we only process JSON files
                file_path = os.path.join(self.output_dir, file)
                with open(file_path, 'r') as f:
                    message_data = json.load(f)
                    history.append(Message(message_data['role'], message_data['content']))
        return history

class TripletExtractor:
    def __init__(self,
                 api_key,
                 model="gpt-3.5-turbo",
                 template_path=LOCAL_RAW_DATA_PATH / 'prompt-templates/triplet-extraction.txt',
                 debug=False): 
        
        self.model = model
        self.debug = debug
        openai.api_key = api_key
        
        with open(file=template_path, encoding='utf8') as fp:
            self.template = fp.read()

    def extract(self, input_dialogue):
        # Prepare the message format for GPT-3.5-turbo model
        
        input_dialogue_str = json.dumps(input_dialogue, indent=2)
        
        messages = [{"role": "system", "content": self.template.format(input_dialogue=input_dialogue_str)}]

        # If debug mode is enabled, don't call the API
        if self.debug:
            return [{'x': 'x_DEBUG', 'x_type': 'type_DEBUG', 'y': 'y_DEBUG', 'y_type': 'type_DEBUG', 'r': 'r_DEBUG'}]

        # Generate response
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            max_tokens=500,  # adjust according to your needs
        )

        # Extract the entities and relations from the response
        relationships = literal_eval(response['choices'][0]['message']['content'])
        return relationships


class ChatGPT:
    def __init__(self,
                 api_key,
                 model="gpt-3.5-turbo",
                 debug=False,
                 output_dir=LOCAL_RAW_DATA_PATH / 'dialogue_logs',
                 bot_name='Adele',
                 user_name='Hilde'):
        
        self.model = model
        self.debug = debug
        openai.api_key = api_key
        self.bot_name = bot_name
        self.user_name = user_name
        self.relationship_extractor = TripletExtractor(api_key=api_key, model=model, debug=debug)
        self.graph_persister = DialogueGraphPersister('chatgpt_pipeline')  
        self.dialogue_logger = DialogueLogger(output_dir)  # Add this line to instantiate the DialogueLogger

        self.load_chat_history()

        # Define your prompt templates
        self.prompt_templates = [
            f"Hi, {self.user_name}, it's {self.bot_name} again! Can you talk now? I wanted to know how your back is doing.",
            f"Hello, {self.user_name}, {self.bot_name} here! Are you free to talk now? I wanted to know how you're feeling about your discussion with your daughter.",
            # f"Hi, {self.user_name}, it's {self.bot_name} again! Can you talk now? I wanted to share some tips I read about having a happier day!",
        ]
        
        if not self.history:
            with open(LOCAL_RAW_DATA_PATH / "prompt-templates/chat-pre-prompt-002.txt", encoding='utf8') as fp:
                preprompt = fp.read().format(bot_name=self.bot_name, user_name=self.user_name)
            self.add_and_log_message('system', preprompt)

    def load_chat_history(self):
        self.history = self.dialogue_logger.load_chat_history()

    def add_and_log_message(self, role, content):
        message = Message(role, content)
        self.history.append(message)
        self.dialogue_logger.save_message(message, len(self.history))
        return message

    def generate_and_add_response(self, user_input):
        # If debug mode is enabled, don't call the API
        if self.debug:
            return self.add_and_log_message("system", "Debugging message")

        # Generate chatbot response
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[msg.to_dict() for msg in self.history],
            max_tokens=60,
        )

        # Extract the chatbot's message from the response
        chatbot_message = response['choices'][0]['message']['content']

        return self.add_and_log_message("system", chatbot_message)

    def format_role(self, role):
        if role == 'system':
            return self.bot_name
        else:
            return self.user_name

    def dump_to_neo4j(self, dialogue, predicted_relations):
        self.graph_persister.process_dialogue(dialogue, predicted_relations)
        self.graph_persister.close_connection()

    def extract_triplets(self, n_last_turns=3, dump_graph=True):
        # Prepare the last n turns as a list
        last_n_turns = [f"{self.format_role(msg.role)}: {msg.content}" for msg in self.history[1:][-n_last_turns:]]

        # Extract relationships
        relationships = self.relationship_extractor.extract(last_n_turns)
        
        if dump_graph and relationships:
            self.dump_to_neo4j(last_n_turns, relationships)
        return relationships
    
    def start_conversation(self):
        print("DEBUG=",self.debug)
        initial_prompt = random.choice(self.prompt_templates)
        self.add_and_log_message("system", initial_prompt)

        print(f"{self.bot_name}: {initial_prompt}")

        # Conversation loop
        while True:
            user_input = input(f"{self.user_name}: ")

            # Exit conversation if user types 'quit'
            if user_input.lower() == 'quit':
                break

            self.add_and_log_message("user", user_input)
            
            # Analyze the conversation history after each user input
            relationships = self.extract_triplets()
            # print(f"# EXTRACTED TRIPLETS: {relationships}")
            
            response = self.generate_and_add_response(user_input)
            print(f"{self.bot_name}: {response.content}")


def load_chat_history(output_dir, max_files=50):
    files = sorted(os.listdir(output_dir), key=lambda x: os.path.getmtime(os.path.join(output_dir, x))) 
    history = []
    for file in files[-max_files:]:  # Only process the last max_files files
        if file.endswith('.json'):  # To ensure we only process JSON files
            file_path = os.path.join(output_dir, file)
            with open(file_path, 'r') as f:
                message_data = json.load(f)
                history.append(Message(message_data['role'], message_data['content']))
    return history


app = Flask(__name__)
@app.route('/')
def home():
    output_dir = LOCAL_RAW_DATA_PATH / 'dialogue_logs'
    history = load_chat_history(output_dir, max_files=50)
    # Format history so that it can be easily processed in the template
    formatted_history = [{"role": message.role, "content": message.content} for message in history]
    return render_template("chat.html", history=formatted_history)

@app.route('/get')
def get_bot_response():
    user_input = request.args.get('msg')
    debug_mode = request.args.get('debug') == 'true'  # Get debug mode from the URL parameters
    chat_gpt = ChatGPT(OPENAI_API_KEY, debug=debug_mode)
    chat_gpt.add_and_log_message("user", user_input)

    # Extract triplets
    relationships = chat_gpt.extract_triplets()

    # Here, you could add your code to dump the relationships into the database, a file, or whatever you choose
    response = chat_gpt.generate_and_add_response(user_input)
    return str(response.content)

@app.route('/proactive')
def get_proactive_response():
    debug_mode = request.args.get('debug') == 'true'  # Get debug mode from the URL parameters
    chat_gpt = ChatGPT(OPENAI_API_KEY, debug=debug_mode)

    # Generate a proactive message from the system (Adelle)
    initial_prompt = random.choice(chat_gpt.prompt_templates)
    chat_gpt.add_and_log_message("system", initial_prompt)

    return str(initial_prompt)

if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    app.run(host='0.0.0.0', port=8080)  # You can use whatever host or port you want
