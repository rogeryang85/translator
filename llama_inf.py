from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

# model = AutoModelForCausalLM.from_pretrained("trl/examples/scripts/output")
model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

newstr = "I have the pleasure to invite the European Community to be represented as a Special Delegation at the Diplomatic Conference on the Protection of Broadcasting Organizations."
sequences = pipeline(f'Please translate the following segment to French: [{newstr}]. Provide just the translated french text, without additional comments or explanations.\n', do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
