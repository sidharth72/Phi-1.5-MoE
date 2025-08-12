import torch, gc
from transformers import AutoTokenizer
from phi_dense_model import load_pretrained as load_dense
from phi_moe_model  import load_pretrained_moe

text = "Hello, how are you?"
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
tokens = tokenizer(text, return_tensors="pt").input_ids

# 2) Run the *dense* model, save its output
dense = load_dense("microsoft/phi-1_5", device="cpu").eval()
with torch.no_grad():
    out_dense = dense(tokens)        # [1,16,V]
torch.save(out_dense.cpu(), "dense_out.pt")
del dense, out_dense
gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

# 3) Load & run your *MoE* model
moe = load_pretrained_moe(
    "microsoft/phi-1_5",
    clustered_ckpt="models/phi_1_5_clustered.pth",
    num_experts=4,
    top_k=2,
    device="cpu"
).eval()

with torch.no_grad():
    out_moe = moe(tokens)

# 4) Load the saved dense output and compare
ref = torch.load("dense_out.pt")
diff = (ref - out_moe).abs().max().item()
print(f"max absolute diff = {diff:.3e}")
assert diff < 1e-3, "MoE≠Dense!"
print("✅ single-expert MoE matches dense")