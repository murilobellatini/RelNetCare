import json
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4")

# Load JSON file
test_file_path = "/home/murilo/RelNetCare/data/processed/dialog-re-37cls-with-no-relation-llama-clsTskOnl/dialog-re-37cls-with-no-relation-llama-clsTskOnl-test.json"
with open(test_file_path, "r") as f:
    data = json.load(f)

# Initialize token count
total_token_count = 0


# Iterate through each conversation
for d in data:
    conversation = d['conversations']
    human_reply = conversation[0]["value"]
    gpt_reply = conversation[1]["value"]
    
    # Combine human and gpt replies
    combined_reply = human_reply + " " + gpt_reply
    
    # Count tokens
    token_count = sum(1 for _ in enc.encode(combined_reply))
    total_token_count += token_count
# Placeholder for token prices
token_price_gpt3_5 = 0.002 / 1000  # Example price for GPT-3.5
token_price_gpt4 = 0.06 / 1000  # Example price for GPT-4

# Compute the cost for GPT-3.5
estimated_cost_gpt3_5 = total_token_count * float(token_price_gpt3_5)

# Compute the cost for GPT-4
estimated_cost_gpt4 = total_token_count * float(token_price_gpt4)

# Print out the results
data_name = '/'.join(test_file_path.split('/')[-2:])
print(f"\n# Data set: {data_name}")
print(f"Sample count:           {len(data):,}")
print(f"Total token count:      {total_token_count:,}")
print(f"Estimated cost GPT-3.5: {estimated_cost_gpt3_5:.2f} USD")
print(f"Estimated cost GPT-4:   {estimated_cost_gpt4:.2f} USD")

