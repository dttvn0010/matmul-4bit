from model import QwenModel
from transformers import AutoTokenizer

def load_model(model_path):
    model = QwenModel.from_pretrained(model_path, device='cuda')
    model.eval()
    return model

def generate(model: QwenModel, tokenizer: AutoTokenizer, prompts: list[str], max_gen_len: int):
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
  
    input_ids = tokenizer(texts)['input_ids']
    output_ids = model.generate(input_ids, max_gen_len, 1.0)
    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

prompts = [
    "What is the sum of 2 and 3?",
    "What is the color of the sky?",
]

# #### download model from huggingface
# from huggingface_hub import snapshot_download 
# snapshot_download("Qwen/Qwen2.5-3B-Instruct-AWQ", local_dir="./Qwen2.5-3B-Instruct-AWQ")

model_path = './Qwen2.5-3B-Instruct-AWQ'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = load_model(model_path)

responses = generate(model, tokenizer, prompts, 256)

for response in responses:
    print("---------------------------------------------------------------------")
    print(response)
