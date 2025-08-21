from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import torch
from peft import PeftModel,LoraConfig,get_peft_model,TaskType
import os
from tqdm import tqdm
from transformers import Seq2SeqTrainingArguments,Seq2SeqTrainer,DataCollatorForSeq2Seq
import numpy as nl
import accelerate
import bitsandbytes
from transformers import BitsAndBytesConfig
class Config:
    def __init__(self):
        self.model_name = "Helsinki-NLP/opus-mt-en-hi"
        self.num_train_epochs = 3
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.max_length = 512
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.01
        self.warmup_steps = 100
        self.gradient_accumulation_steps = 2
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.data_limit = 10000
        self.dataset_name = "cfilt/iitb-english-hindi"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fp16 = True if torch.cuda.is_available() else False
        self.evaluation_strategy = "epoch"
        self.save_strategy = "epoch"
        self.max_target_length = 512
class Datapreprocessor:
    def __init__(self,config:Config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        def load_dataset(self):
            self.dataset = load_dataset(config.dataset_name, split="train[:self.config.data_limit]")
            self.dataset = self.dataset.train_test_split(test_size=0.1, seed=42)
            return self.dataset
        def preprocess(self,examples):
            inputs = [ex['en'] for ex in examples['translation']]
            targets = [ex['hi'] for ex in examples['translation']]
            model_inputs = self.tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
            labels = self.tokenizer(text_target=targets, max_length=512, truncation=True, padding="max_length")
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        def get_dataset(self):
            return self.dataset.map(self.preprocess,batched=True,remove_columns=self.dataset['train'].column_names)

class ModelManager:
    def __init__(self,config:Config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.peft_model = None
    def setup_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16

        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        self.peft_model = get_peft_model(self.model,lora_config)
        self.peft_model.print_trainable_parameters()
        return self.peft_model, self.tokenizer
    def save_model(self,output_dir):
        os.makedirs(output_dir, exist_ok=True) 
        self.peft_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")
    def load_model(self,model_path):
        self.peft_model = PeftModel.from_pretrained(self.model,model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        return self.peft_model, self.tokenizer
class TrainerPipeline:
    def __init__(self,config:Config):
        self.config = config
        self.model_manager = ModelManager(config)
        self.data_preprocessor = Datapreprocessor(config)
        os.makedirs(self.config.output_dir, exist_ok=True)
    def compute_metrics(self,eval_preds):
        predictions,references = eval_preds
        tokenizer = self.data_preprocessor.tokenizer
        bleu = evaluate.load("sacrebleu")
        chrf = evaluate.load("chrf")
        ter = evaluate.load("ter")
        meteor = evaluate.load("meteor")
        preds = [pred.strip() for pred in predictions]
        labels = [[label.strip()] for label in references]
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        results = {}
        results["bleu"] = bleu.compute(predictions=decoded_preds, references=decoded_labels)["score"]
        results["chrf"] = chrf.compute(predictions=decoded_preds, references=decoded_labels)["score"]
        results["ter"] = ter.compute(predictions=decoded_preds, references=decoded_labels)["score"]
        results["meteor"] = meteor.compute(predictions=decoded_preds, references=decoded_labels)["meteor"]
        return results
    def train(self):
        tokenized_dataset = self.data_preprocessor.get_dataset()
        model,tokenizer = self.model_manager.setup_model_and_tokenizer()
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            eval_strategy=self.config.evaluation_strategy,
            save_strategy=self.config.save_strategy,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_train_epochs,
            weight_decay=0.01,
            warmup_steps=self.config.warmup_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.config.fp16,
            predict_with_generate=True,
            max_length=self.config.max_length,
            max_target_length=self.config.max_target_length,
            save_total_limit = 3,
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        trainer.train()
        trainer.evaluate()
        self.model_manager.save_model(self.config.output_dir)
        return trainer
if __name__ == "__main__":
    config = Config()
    trainer_pipeline = TrainerPipeline(config)
    trainer ,eval_results= trainer_pipeline.train()
    print("Training complete. Model saved to:", config.output_dir)