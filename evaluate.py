import evaluate
from datasets import load_dataset
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
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

class EvaluateMetrics:
    def __init__(self, dataset, model, tokenizer):
        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.bleu = evaluate.load("sacrebleu")
        self.chrf = evaluate.load("chrf")
        self.ter = evaluate.load("ter")
        self.meteor = evaluate.load("meteor")

    def generate_predictions(self, max_examples=None):
        preds, refs = [], []
        limit = len(self.dataset) if max_examples is None else min(max_examples, len(self.dataset))
        for i in tqdm(range(limit)):
            ex = self.dataset[i]
            refs.append(ex["translation"]["hi"])
            inputs = self.tokenizer(ex["translation"]["en"], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
            preds.append(self.tokenizer.decode(out[0], skip_special_tokens=True))
        return preds, refs

    def compute_metrics(self, preds, refs):
        refs_list = [[r] for r in refs]
        return {
            "bleu": self.bleu.compute(predictions=preds, references=refs_list)["score"],
            "chrf": self.chrf.compute(predictions=preds, references=refs_list)["score"],
            "ter": self.ter.compute(predictions=preds, references=refs_list)["score"],
            "meteor": self.meteor.compute(predictions=preds, references=refs_list)["meteor"],
        }

if __name__ == "__main__":
    dataset = load_dataset("cfilt/iitb-english-hindi", split="test[:200]")
    evaluator = EvaluateMetrics(dataset, model, tokenizer)
    preds, refs = evaluator.generate_predictions()
    metrics = evaluator.compute_metrics(preds, refs)
    print("Evaluation Metrics:", metrics)