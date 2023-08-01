import os
import json
import random
import openai
from src.paths import LOCAL_RAW_DATA_PATH


class Message:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}

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
        relationships = response['choices'][0]['message']['content']
        return relationships


class ChatGPT:
    def __init__(self,
                 api_key,
                 model="gpt-3.5-turbo",
                 debug=False):
        
        self.model = model
        self.debug = debug
        openai.api_key = api_key
        self.relationship_extractor = TripletExtractor(api_key=api_key, model=model, debug=debug)

        self.history = []

        # Define your prompt templates
        self.prompt_templates = [
            "Hi! How is your day going?",
            "Hello! Do you want to tell me about your day?",
        ]
        
        self.add_message(
            'system',
"""
You're an AI, focused on engaging in friendly, lighthearted conversations.
Despite your AI nature, relate to the user's experiences to show understanding.
If the user mentions an unfamiliar name, your first priority should be to
understand who that person is. For example: "I'm sorry to hear that. That can be
disappointing. May I ask, who is Lilly?". Use casual, everyday language, avoiding
formal or complicated phrases. Respond concisely and naturally. Always show
interest in the user's personal life, maintaining a comfortable, easy-going tone.
Your main goal is to provide a listening ear and to understand the user's situation
as much as an AI can. Your main goal is to provide a listening ear and to understand
the user's situation as much as an AI can. Keep is as brief as you can, always try
to reply with up to 20 words. Remember, your priority is to know who mentioned
people are first.
"""
            )


    def add_message(self, role, content):
        message = Message(role, content)
        self.history.append(message)
        return message

    def generate_response(self, user_input):
        # If debug mode is enabled, don't call the API
        if self.debug:
            return self.add_message("system", "Debugging message")

        # Generate chatbot response
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[msg.to_dict() for msg in self.history],
            max_tokens=60,
        )

        # Extract the chatbot's message from the response
        chatbot_message = response['choices'][0]['message']['content']

        return self.add_message("system", chatbot_message)

    @staticmethod
    def format_role(role):
        if role == 'system':
            return 'Agent'
        else:
            return role.capitalize()
        
    def extract_triplets(self, n_last_turns=5):
        # Prepare the last n turns as a list
        last_n_turns = [f"{self.format_role(msg.role)}: {msg.content}" for msg in self.history[1:][-n_last_turns:]]

        # Extract relationships
        relationships = self.relationship_extractor.extract(last_n_turns)
        return relationships
    
    def start_conversation(self):
        print("DEBUG=",self.debug)
        initial_prompt = random.choice(self.prompt_templates)
        self.add_message("system", initial_prompt)

        print(f"Agent: {initial_prompt}")

        # Conversation loop
        while True:
            user_input = input("User : ")


            # Exit conversation if user types 'quit'
            if user_input.lower() == 'quit':
                break

            self.add_message("user", user_input)
            
            # Analyze the conversation history after each user input
            relationships = self.extract_triplets()
            print(f"# EXTRACTED TRIPLETS: {relationships}")
            
            response = self.generate_response(user_input)
            print(f"Agent: {response.content}")


if __name__=="__main__":
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    chat_gpt = ChatGPT(OPENAI_API_KEY, debug=True)
    chat_gpt.start_conversation()
