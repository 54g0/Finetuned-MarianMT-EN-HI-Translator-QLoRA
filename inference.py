import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import PeftModel
model_name = "Helsinki-NLP/opus-mt-en-hi"
peft_model_path = "output"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, peft_model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_model_path)

class Inference:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def predict(self, text: str):
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(text.strip(), return_tensors="pt").to(self.model.device)
            out = self.model.generate(**inputs, max_length=512)
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    user_input = input("Enter your input text: ")
    engine = Inference(model, tokenizer)
    result = engine.predict(user_input)
    print("Output:", result)
