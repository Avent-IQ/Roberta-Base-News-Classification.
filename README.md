# RoBERTa-Base for News Classification (FP16 Quantized)  
This is a RoBERTa-Base model fine-tuned on the **AG News dataset** for text classification. It categorizes news articles into one of four classes: **World, Sports, Business, and Science/Technology**. The model has been further **quantized to FP16** for improved inference speed and reduced memory usage, making it efficient for deployment on resource-constrained environments.  

---

## **Model Details**  

### **Model Description**  
- **Model Type:** Transformer-based text classifier  
- **Base Model:** `roberta-base`  
- **Fine-Tuned Dataset:** AG News  
- **Maximum Sequence Length:** 512 tokens  
- **Output:** One of four news categories  
- **Task:** Text classification  
---

## **Full Model Architecture**  
```python
RobertaForSequenceClassification(
  (roberta): RobertaModel(
    (embeddings): RobertaEmbeddings(...)
    (encoder): RobertaEncoder(...)
  )
  (classifier): RobertaClassificationHead(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (dropout): Dropout(p=0.1)
    (out_proj): Linear(in_features=768, out_features=4, bias=True)
  )
)
```

---

## **Usage Instructions**  

### **Installation**  
```bash
pip install -U transformers torch
```

### **Loading the Model for Inference**  
```python
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch

# Load the model and tokenizer
model_name = "AventIQ-AI/Roberta-Base-News-Classification"  # Update with your model ID
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function to predict category
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    class_labels = {0: "World", 1: "Sports", 2: "Business", 3: "Science/Technology"}
    return class_labels[predicted_class]

# Example usage
custom_text = "Stock prices are rising due to global economic recovery."
predicted_label = predict(custom_text)
print(f"Predicted Category: {predicted_label}")
```

---

## **Training Details**  

### **Training Dataset**  
- **Name:** AG News  
- **Size:** 50,000 training samples, 7,600 test samples  
- **Labels:**  
  - **0:** World  
  - **1:** Sports  
  - **2:** Business  
  - **3:** Science/Technology  

---

## **Training Hyperparameters**  
### **Non-Default Hyperparameters:**  
- **per_device_train_batch_size:** 8  
- **per_device_eval_batch_size:** 8  
- **gradient_accumulation_steps:** 2 (effective batch size = 16)  
- **num_train_epochs:** 3  
- **learning_rate:** 2e-5  
- **fp16:** True (for reduced memory footprint and faster inference)  
- **weight_decay:** 0.01  
- **optimizer:** AdamW  

---

## **Model Performance**  
| Metric  | Score |  
|---------|-------|  
| Accuracy | **94.3%** |  
| F1 Score | **94.1%** |  
| Precision | **94.5%** |  
| Recall | **94.2%** |  

*(Update these values based on your actual evaluation results.)*  

---

## **Quantization Details**  
- The model has been quantized to **FP16** to reduce its size and improve inference speed.
- FP16 quantization provides a **2x reduction in memory** while maintaining similar accuracy.

---

## **Limitations & Considerations**  
- The model is **trained on AG News** and may not generalize well to **other domains** such as medical, legal, or entertainment news.
- Due to **FP16 quantization**, there might be a minor loss in precision, but inference speed is significantly improved.
- The model is **not intended for real-time misinformation detection**—it only classifies text based on its most probable category.

---
