import sys
import torch
from transformers import AutoTokenizer
# from phi_dense_model import load_pretrained
from phi_moe_model import load_pretrained_moe, load_dense_as_moe

# 1. Load model and tokenizer
print("Loading model and tokenizer...")
model = load_pretrained_moe(clustered_ckpt="models/phi_1_5_clustered.pth", num_experts=4, top_k=2).eval()

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

while True:
    # 2. Get the prompt from the command line
    prompt = input("Enter a prompt: ")

    # 3. Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # 4. Autoregressive generation
    print("\n--- Generating Text ---")
    generated_ids = input_ids
    max_new_tokens = 15

    for _ in range(max_new_tokens):
        with torch.no_grad():
            # The model expects input_ids of shape [batch_size, seq_len]
            outputs = model(generated_ids)

            # Get the logits for the last token
            next_token_logits = outputs[:, -1, :]

            # Greedy decoding: select the token with the highest probability
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append the new token to the sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

            # Stop if EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    # 5. Decode and print the generated text
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n--- Generated Text ---")
    print(text)