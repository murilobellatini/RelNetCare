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
        
        self.api_key = api_key
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
            return "Debugging message"

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
        
        self.api_key = api_key
        self.model = model
        self.debug = debug
        self.history = []
        self.add_message(
            'system',
"""
You're an AI designed for empathetic, friendly conversation.
Use "experiences" to show understanding, acknowledging you're
an AI. For instance: "I'm sorry to hear that. It can be
disappointing when people don't get back to us. Could you tell
me more about Lilly?". Prompt for clarifications when you encounter
unfamiliar details, and display interest in the user's personal
life. Keep the conversation lighthearted and approachable. Respond
briefly and naturally, using everyday language. Your main goal is
to provide a listening ear and to understand the user's situation
as much as an AI can.
"""
            )
        openai.api_key = api_key
        self.relationship_extractor = TripletExtractor(api_key=api_key, model=model, debug=debug)

        # Define your prompt templates
        self.prompt_templates = [
            "Hi! How is your day going?",
            "Hello! Do you want to tell me about your day?",
        ]

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
    chat_gpt = ChatGPT(OPENAI_API_KEY, debug=False)
    chat_gpt.start_conversation()
