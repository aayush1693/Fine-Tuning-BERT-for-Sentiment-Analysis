# Fine-Tuning BERT for Sentiment Analysis

## Description
This repository contains code for fine-tuning a BERT (Bidirectional Encoder Representations from Transformers) model for sentiment analysis using the Huggingface transformers library.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Inference](#inference)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/aayush1693/Fine-Tuning-BERT-for-Sentiment-Analysis.git
   ```

2. Navigate to the project directory:
   ```sh
   cd Fine-Tuning-BERT-for-Sentiment-Analysis
   ```

3. Install the required dependencies:
   ```sh
   pip install transformers datasets matplotlib pandas scikit-learn
   ```

## Usage

1. Prepare your dataset and ensure it is in the required format. This project uses the SST-2 dataset from Huggingface.

2. Run the training script. Replace `<path_to_data>` and `<path_to_output>` with your actual data and output directories:
   ```sh
   python fine_tune_bert_sentiment.py --data_dir <path_to_data> --output_dir <path_to_output>
   ```

3. Evaluate the model. Replace `<path_to_model>` with your actual model directory:
   ```sh
   python evaluate.py --model_dir <path_to_model>
   ```

## Training and Evaluation

The project includes a Jupyter notebook (`fine_tune_bert_sentiment.ipynb`) that demonstrates the entire process of fine-tuning a BERT model for sentiment analysis. Below are the key steps:

1. **Install Dependencies**:
   ```python
   !pip install transformers datasets matplotlib
   ```

2. **Load and Prepare Dataset**:
   ```python
   from datasets import load_dataset
   dataset = load_dataset('glue', 'sst2')
   train_data = dataset['train']
   test_data = dataset['validation']
   ```

3. **Tokenize Data**:
   ```python
   from transformers import BertTokenizer
   tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   def tokenize(batch):
       return tokenizer(batch['sentence'], padding=True, truncation=True)
   train_data = train_data.map(tokenize, batched=True, batch_size=len(train_data))
   test_data = test_data.map(tokenize, batched=True, batch_size=len(test_data))
   ```

4. **Load Pre-trained BERT Model**:
   ```python
   from transformers import BertForSequenceClassification
   model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
   ```

5. **Define Metrics and Evaluation Function**:
   ```python
   from sklearn.metrics import accuracy_score, precision_recall_fscore_support
   def compute_metrics(pred):
       labels = pred.label_ids
       preds = pred.predictions.argmax(-1)
       precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
       acc = accuracy_score(labels, preds)
       return {
           'accuracy': acc,
           'f1': f1,
           'precision': precision,
           'recall': recall
       }
   ```

6. **Setup Training Arguments**:
   ```python
   from transformers import TrainingArguments
   training_args = TrainingArguments(
       output_dir='./results',          
       evaluation_strategy="epoch",     
       num_train_epochs=3,              
       per_device_train_batch_size=16,  
       per_device_eval_batch_size=64,   
       warmup_steps=500,                
       weight_decay=0.01,               
       logging_dir='./logs',            
       logging_steps=10,                
       load_best_model_at_end=True,     
       save_strategy="epoch",           
   )
   ```

7. **Initialize Trainer**:
   ```python
   from transformers import Trainer
   trainer = Trainer(
       model=model,                         
       args=training_args,                  
       train_dataset=train_data,            
       eval_dataset=test_data,              
       compute_metrics=compute_metrics,     
   )
   ```

8. **Train the Model and Visualize Training Loss**:
   ```python
   train_result = trainer.train()
   model.save_pretrained('./fine_tuned_bert')
   tokenizer.save_pretrained('./fine_tuned_bert')
   ```

9. **Evaluate the Model**:
   ```python
   eval_result = trainer.evaluate()
   for key, value in eval_result.items():
       print(f"{key}: {value:.4f}")
   ```

10. **Visualize Training and Evaluation Metrics**:
    ```python
    metrics = eval_result
    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_values = [metrics['eval_accuracy'], metrics['eval_precision'], metrics['eval_recall'], metrics['eval_f1']]
    plt.bar(metric_names, metric_values, color=['blue', 'orange', 'green', 'red'])
    plt.title('Evaluation Metrics')
    plt.ylim(0, 1)
    plt.show()
    ```

## Inference

You can perform inference on new sentences using the fine-tuned model:

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

fine_tuned_model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert')
fine_tuned_tokenizer = BertTokenizer.from_pretrained('./fine_tuned_bert')

sentence = "This movie is fantastic!"
inputs = fine_tuned_tokenizer(sentence, return_tensors="pt")
outputs = fine_tuned_model(**inputs)
prediction = torch.argmax(outputs.logits, dim=-1)
label = 'positive' if prediction == 1 else 'negative'
print(f"Sentiment: {label}")
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact Information

For any inquiries or issues, please contact [aayush1693](https://github.com/aayush1693).
