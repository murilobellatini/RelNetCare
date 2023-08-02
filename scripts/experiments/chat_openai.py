import os
import time
import json
import random
import openai
import shutil
import pickle
from ast import literal_eval
from datetime import datetime
from flask import Flask, render_template, request

from src.processing.neo4j_operations import DialogueGraphPersister
from src.paths import LOCAL_RAW_DATA_PATH

USER_NAME = 'Hilde'
BOT_NAME = 'Adele'
DATASET_NAME = 'chatgpt_pipeline'

class Message:
    def __init__(self, role, content, timestamp=None):
        self.role = role
        self.content = content
        if not timestamp:
            timestamp = time.time()
        self.timestamp = timestamp

    def to_dict(self, drop_timestamp=False):
        base_dict = {"role": self.role, "content": self.content}
        if not drop_timestamp:
            base_dict["timestamp"] = self.timestamp
        return base_dict
    
class DialogueLogger:
    def __init__(self, output_dir, dataset_name=DATASET_NAME):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.persister = DialogueGraphPersister("my_dataset")

    def save_message(self, message: Message, turn):
        message_data = {
            "turn": turn,
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,  # Timestamp in seconds since the Epoch
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
                    history.append(Message(message_data['role'], message_data['content'], message_data['timestamp']))
        return history

    def archive_dialogue_logs(self):
        # Create a folder to store archived logs
        archive_folder = os.path.join(self.output_dir, "archive")
        os.makedirs(archive_folder, exist_ok=True)

        # Define the name of the archive folder for this specific dialogue
        timestamp = int(time.time())  # Timestamp in seconds since the Epoch
        human_readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H-%M-%S')
        archive_subfolder = os.path.join(archive_folder, f"{USER_NAME}_{BOT_NAME}_{human_readable_time}")

        # Create a folder for the archive of this dialogue
        os.makedirs(archive_subfolder, exist_ok=True)

        # Move all dialogue log files to the archive folder
        files = os.listdir(self.output_dir)
        for file in files:
            if file.endswith('.json'):  # To ensure we only process JSON files
                file_path = os.path.join(self.output_dir, file)
                new_file_path = os.path.join(archive_subfolder, file)
                os.rename(file_path, new_file_path)
        self.persister.archive_and_clean(os.path.join(archive_subfolder, "neo4j_dump"))
        self.persister.close_connection()
    
    def get_archive_folders(self):
        archive_folder = os.path.join(self.output_dir, "archive")

        try:
            return os.listdir(archive_folder)
        except FileNotFoundError:
            return []    
    
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

class OpenerGenerator:
    def __init__(self, user_name, bot_name, state_path='./opener_state.pkl'):
        self.user_name = user_name
        self.bot_name = bot_name
        self.state_path = state_path

        self.greetings = [
            f"Hello, {self.user_name}, it's {self.bot_name} here!",
            f"Hi, {self.user_name}, this is {self.bot_name}!",
            f"Good day, {self.user_name}! It's {self.bot_name} here.",
            f"{self.bot_name} here, hi {self.user_name}!"
        ]

        self.availability_requests = [
            "Do you have a moment for a chat?",
            "Can we talk now?",
            "Is now a good time for a chat?",
            "Are you free to talk now?",
            "Could we have a quick chat?"
        ]

        self.all_topic_introductions = [
            "I was curious about how you're managing your medications.",
            "I was wondering, how large is your family?",
            "I was wondering, what hobbies do you enjoy?",
            "I was curious, if you have any favorite interests.",
            "I was wondering, do you like reading?",
            "I was wondering, what kinds of songs do you like?",
            "I was wondering, if you have any cherished memories. I'd love to know them!",
            "I was curious, do you have any pets?",
            "I wanted to know, how is your back doing?"
        ]
        
        # Load or initialize topic introductions
        try:
            with open(self.state_path, 'rb') as f:
                self.topic_introductions = pickle.load(f)
        except (FileNotFoundError, EOFError):
            self.topic_introductions = self.all_topic_introductions.copy()

    def save_state(self):
        with open(self.state_path, 'wb') as f:
            pickle.dump(self.topic_introductions, f)

    def generate_opener(self):
        greeting = random.choice(self.greetings)
        availability_request = random.choice(self.availability_requests)

        if not self.topic_introductions:
            # all topics have been used, so repopulate the list
            self.topic_introductions = self.all_topic_introductions.copy()

        topic_introduction = random.choice(self.topic_introductions)
        self.topic_introductions.remove(topic_introduction)
        
        # Save the current state
        self.save_state()

        return f"{greeting} {availability_request} {topic_introduction}"

class ChatGPT:
    def __init__(self,
                 api_key,
                 model="gpt-3.5-turbo",
                 debug=False,
                 output_dir=LOCAL_RAW_DATA_PATH / 'dialogue_logs',
                 bot_name=BOT_NAME,
                 user_name=USER_NAME,
                 dataset_name=DATASET_NAME):
        
        self.model = model
        self.debug = debug
        openai.api_key = api_key
        self.bot_name = bot_name
        self.user_name = user_name
        self.relationship_extractor = TripletExtractor(api_key=api_key, model=model, debug=debug)
        self.graph_persister = DialogueGraphPersister(dataset_name)  
        self.dialogue_logger = DialogueLogger(output_dir)  # Add this line to instantiate the DialogueLogger

        self.load_chat_history()

        # Initialize the OpenerGenerator
        self.opener_generator = OpenerGenerator(self.user_name, self.bot_name, output_dir / "opener_state.pkl")

        
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
            messages=[msg.to_dict(drop_timestamp=True) for msg in self.history],
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
        initial_prompt = self.opener_generator.generate_opener()
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

    def reload_from_archive(self, archive_subfolder):
        archive_path = os.path.join(self.dialogue_logger.output_dir, "archive", archive_subfolder)
        output_dir = self.dialogue_logger.output_dir

        # Step 1: Load the chat history
        self.history = []
        for file in os.listdir(archive_path):
            if file.endswith('.json'):
                file_path = os.path.join(archive_path, file)
                with open(file_path, 'r') as f:
                    message_data = json.load(f)
                    self.history.append(Message(message_data['role'], message_data['content'], message_data['timestamp']))

        # Step 2: Delete all JSON files in the output directory
        for file in os.listdir(output_dir):
            if file.endswith('.json'):
                file_path = os.path.join(output_dir, file)
                os.remove(file_path)

        # Step 3: Copy all files from the archive folder to the output directory
        for file in os.listdir(archive_path):
            if file.endswith('.json'):
                src_path = os.path.join(archive_path, file)
                dst_path = os.path.join(output_dir, file)
                shutil.copy(src_path, dst_path)

        # Step 4: Load the graph data
        self.graph_persister.load_archived_data(archive_path)


def load_chat_history(output_dir, max_files=50):
    files = sorted(os.listdir(output_dir), key=lambda x: os.path.getmtime(os.path.join(output_dir, x))) 
    history = []
    for file in files[-max_files:]:  # Only process the last max_files files
        if file.endswith('.json'):  # To ensure we only process JSON files
            file_path = os.path.join(output_dir, file)
            with open(file_path, 'r') as f:
                message_data = json.load(f)
                if message_data.get('turn') == 1:
                    continue # Skip preprompt message
                history.append(Message(message_data['role'], message_data['content'], message_data['timestamp']))
    return history


app = Flask(__name__)
@app.route('/')
def home():
    output_dir = LOCAL_RAW_DATA_PATH / 'dialogue_logs'
    history = load_chat_history(output_dir, max_files=50)
    # Format history so that it can be easily processed in the template
    formatted_history = [message.to_dict() for message in history]
    return render_template("chat.html", history=formatted_history, bot_name=BOT_NAME, user_name=USER_NAME)

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
    
    # Generate a proactive message from the system (Adele)
    initial_prompt = chat_gpt.opener_generator.generate_opener()
    chat_gpt.add_and_log_message("system", initial_prompt)

    return str(initial_prompt)

@app.route('/archive')
def archive_logs():
    chat_gpt = ChatGPT(OPENAI_API_KEY, debug=False)
    chat_gpt.dialogue_logger.archive_dialogue_logs()
    return "Logs have been archived successfully."

@app.route('/archives')
def list_archives():
    dialogue_logger = DialogueLogger(LOCAL_RAW_DATA_PATH / 'dialogue_logs')
    archive_folders = dialogue_logger.get_archive_folders()
    return json.dumps(archive_folders)

@app.route('/load_archive/<archive_name>')
def load_archive(archive_name):
    chat_gpt = ChatGPT(OPENAI_API_KEY, debug=False)
    chat_gpt.reload_from_archive(archive_name)
    return f"Archive '{archive_name}' loaded successfully."

if __name__ == "__main__":
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    app.run(host='0.0.0.0', port=8080)  # You can use whatever host or port you want
