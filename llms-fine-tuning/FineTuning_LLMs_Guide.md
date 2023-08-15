# Fine-Tuning LLMs: A Walkthrough 

This guide provides a comprehensive walkthrough on how to fine-tune Large Language Models (LLMs), specifically aiming to mock OpenAI's chat completion endpoint so it aligns seamlessly with ChatGPT. We'll focus on fine-tuning the Vicuna-7b model using Vicuna, which provides better training effects due to its multi-round dialogue corpus. 

## Fine-tuning the Model 

We'll be taking inspiration from the OpenAI ChatCompletion Endpoint, more specifically from [this repository](https://github.com/git-cloner/llama-lora-fine-tuning/tree/main). This approach leverages Vicuna based on the ShareGPT corpus, offering a significant advantage over the single-round dialogue system of Alpaca.

### Steps:

1. **Download Llama Model**: Start by downloading the required model.

2. **Clone Repository**: Clone the `llama-lora-fine-tuning` repository.

3. **Install Dependencies**: Ensure you're using the right version compatibility.

4. **Modify FastChat Dependency**: Use `gitclone` to fetch the `fastchat` dependency. You'll need to adjust the code for the Conversation Template based on the [conversation.py](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py) file.

5. **Convert Model**: Ensure the model is in the correct format for training.

6. **Fine-tuning**: Now, you'll fine-tune your model. Given GPU memory constraints, only a few layers can be fine-tuned. Use the commands below:

```bash
deepspeed fastchat/train/train_lora.py \
    --deepspeed ./deepspeed-config.json \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --model_name_or_path ~/Development/LLM/LLAMA/7B/Transformer/ \
    --data_path ~/Development/LLM/FastChat/data/spice_finetune_dataset_chat_30000_v02.json \
    --fp16 True \
    --output_dir ./output/lora-adapter/spice-adapter_v02 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1\
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1200 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 512 \
    --gradient_checkpointing True
```

7. **Merge Fine-tuned Layers**: After fine-tuning, merge the adjusted layers into the main model to enable quick inference:

```bash
python3 -m fastchat.model.apply_lora --base /path/to/LLM/LLAMA --target /path/to/output/lora-model --lora /path/to/output/lora-adapter
```

Example parameters:

* Base: `/home/ubuntu/Development/LLM/LLAMA/7B/Transformer`
* Target: `/home/ubuntu/Development/LLM/Lora/llama-lora-fine-tuning/output/lora-model/spice-model`
* Lora: `/home/ubuntu/Development/LLM/Lora/llama-lora-fine-tuning/output/lora-adapter/spice-adapter`

## Mocking the OpenAI API with the Fine-tuned Model 

Utilize FastChat, an open platform designed for training, serving, and evaluating LLM-based chatbots, to mock the OpenAI API. The platform also supports OpenAI-compatible RESTful APIs. For more details, refer to [FastChat's repository](https://github.com/lm-sys/FastChat).

### Steps:

1. **Deploy FastChat API Wrapper**: 

```bash
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.model_worker --model-name 'vicuna-7b-v1.1' --model-path ./path/to/Vicuna
python3 -m fastchat.serve.openai_api_server --host localhost --port 8000
```

Example parameters:

* model-path: `./Vicuna/7B/Transformer`

2. **Port Forwarding**: Access the FastChat API locally with:

```bash
ssh -L 8000:localhost:8000 user@your.ip.address.here
```

Example parameters:

* your.ip.address.here: `10.195.2.147`

## Notes:

- Always use `vicuna_v1.1` as the conversation template for fine-tuning.
- Use the endpoint `localhost:8000/worker_get_conversation_template` to verify the conversation template in use.
- For additional context or other approaches, refer to Stanford's [Alpaca-LoRA](https://github.com/tloen/alpaca-lora) model.

---

For a deeper dive, you can explore the referenced repositories:
- [llama-lora-fine-tuning](https://github.com/git-cloner/llama-lora-fine-tuning/tree/main)
- [FastChat](https://github.com/lm-sys/FastChat)
- [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)


# Optional: Improving the FastChat Overwrite Process Using Runtime Overrides


1. **Import the Necessary Components**: Before making any changes, ensure you have the necessary components from the `fastchat` package.
2. **Define the New Conversation Template**: This will allow the model to understand the context and format of a conversation.
3. **Register the New Template**: By registering, the model will use this new conversation template for interactions.

```python
# Assuming you've already installed and set up fastchat...

from fastchat import Conversation, SeparatorStyle, register_conv_template

# Define the new conversation template locally.
new_template = Conversation(
    name="vicuna_v1.1",
    system_message="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=("USER", "ASSISTANT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.ADD_COLON_TWO,
    sep=" ",
    sep2="</s>",
)

# Register/override the new template at runtime.
register_conv_template(new_template, override=True)
```