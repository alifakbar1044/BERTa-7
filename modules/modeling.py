import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback, set_seed
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load Model Base
MODEL_NAME = "flax-community/indonesian-roberta-base"

class SentimentDataset(torch.utils.data.Dataset):
    # Dataset untuk Sentiment Analysis
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        
    # Fungsi untuk mengambil item dari dataset
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    # Fungsi untuk mendapatkan panjang dataset
    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_indoroberta(X_train, y_train, X_val, y_val, hyperparams):
    set_seed(42)
    
    # 1. Tokenisasi
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(list(X_val), truncation=True, padding=True, max_length=128)
    
    train_dataset = SentimentDataset(train_encodings, list(y_train))
    val_dataset = SentimentDataset(val_encodings, list(y_val))
    
    # 2. Siapkan Model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # 3. Training Arguments
    training_args = TrainingArguments(
        output_dir='./results_temp',
        num_train_epochs=hyperparams['epochs'],
        per_device_train_batch_size=hyperparams['batch_size'],
        per_device_eval_batch_size=hyperparams['batch_size'],
        learning_rate=hyperparams['lr'],
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        
        # Evaluasi & Saving
        logging_strategy="epoch",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        
        use_cpu=True if device == "cpu" else False,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # 4. Mulai Training
    trainer.train()
    
    # 5. Return
    return trainer, tokenizer, model