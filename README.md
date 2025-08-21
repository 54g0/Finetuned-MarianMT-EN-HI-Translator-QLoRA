**MarianMT English → Hindi Translator (Fine-tuned with QLoRA)**


-- **Project Overview**

This project demonstrates how to fine-tune a MarianMT model (Helsinki-NLP/opus-mt-en-hi) for English → Hindi translation using QLoRA (Quantized Low-Rank Adaptation).

Compared to the base MarianMT model:

-- Better translation accuracy (higher BLEU, CHRF, METEOR scores)

-- Faster inference due to QLoRA’s efficient parameterization

-- Lower GPU memory usage during training

The fine-tuned model is publicly available on 🤗 Hugging Face Hub:
-> 54gO/marianmt-en-hi-qlora

**Key Features**

Fine-tuned using QLoRA for efficiency

Faster inference than the base MarianMT model(Helsinki-NLP/opus-mt-en-hi)

Evaluated on BLEU, CHRF, METEOR, TER

Fully open-source pipeline: reproducible training, evaluation, and usage


**Results & Benchmarks**

| Metric              | Base Model | Fine-tuned Model   |
| ------------------- | ---------- | -----------------  |
| **BLEU**            | 61.43         | **81.58**       |
| **CHRF**            | 68.11         | **87.07**       |
| **METEOR**          | 0.6           | **0.8**         |
| **TER ↓**           | 22.13         | **16.29**       |
| **Inference Speed** | 1x            | **\~4x faster** |


**Training Details**

Base Model: Helsinki-NLP/opus-mt-en-hi

Fine-tuning method: QLoRA

Dataset: "cfilt/iitb-english-hindi"

Batch size: 16

Epochs: 2

Learning rate: 1e-4

Hardware: [T4 GPU ]

**How to Use as an English to Hindi Translator**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "54gO/marianmt-en-hi-translator"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text = "How are you?"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
Output: "आप कैसे हैं?"
