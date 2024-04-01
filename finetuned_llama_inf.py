from peft import PeftModel
from transformers import LlamaTokenizerFast,PreTrainedTokenizerFast,  AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import transformers
import torch

# model = AutoModelForCausalLM.from_pretrained("trl/examples/scripts/output")
model_name = "meta-llama/Llama-2-7b-chat-hf"
# adapters_name = "/home/ubuntu/trl/examples/scripts/output"
adapters_name = "/home/ubuntu/trl/examples/scripts/llama_model_output"

# tokenizer = AutoTokenizer.from_pretrained(model_name)

m = AutoModelForCausalLM.from_pretrained(
    model_name,
    #load_in_4bit=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)

m = PeftModel.from_pretrained(m, adapters_name)
m = m.merge_and_unload()


# tokenizer = LlamaTokenizer.from_pretrained(model_name)
# tokenizer.bos_token_id = 1
# tokenizer = LlamaTokenizer.from_pretrained(model_name)
# tokenizer = LlamaTokenizerFast(tokenizer_file="/home/ubuntu/trl/examples/scripts/output/tokenizer.json")
tokenizer = LlamaTokenizerFast(tokenizer_file="/home/ubuntu/trl/examples/scripts/llama_model_output/tokenizer.json")

stop_token_ids = [0]

print(f"Successfully loaded the model {model_name} into memory")


pipeline = transformers.pipeline(
    "text-generation",
    model=m,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer=tokenizer,
)
newstr = "I have the pleasure to invite the European Community to be represented as a Special Delegation at the Diplomatic Conference on the Protection of Broadcasting Organizations."
sequences = pipeline(f'Translate from English to french to French: [{newstr}]. Provide just the translated french text, without additional comments or explanations.\n', do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
