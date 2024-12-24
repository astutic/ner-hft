import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import pandas as pd
import numpy as np
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score

def read_conll(file_path):
    sentences, tags = [], []
    current_sent, current_tags = [], []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                token, tag = line.split('\t')
                current_sent.append(token)
                current_tags.append(tag)
            elif current_sent:
                sentences.append(current_sent)
                tags.append(current_tags)
                current_sent, current_tags = [], []
                
    if current_sent:
        sentences.append(current_sent)
        tags.append(current_tags)
    
    return sentences, tags

def tokenize_and_align_labels(examples, tokenizer, label2id):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
            
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def main():
    # Load data
    train_sentences, train_tags = read_conll("data/train.conll")
    
    # Get unique tags
    unique_tags = sorted(list(set(tag for tags in train_tags for tag in tags)))
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    
    # Create dataset
    train_dataset = Dataset.from_dict({
        "tokens": train_sentences,
        "ner_tags": [[tag2id[t] for t in tags] for tags in train_tags]
    })
    
    # Initialize tokenizer and model
    model_name = "bert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(unique_tags)
    )
    
    # Tokenize dataset
    tokenized_dataset = train_dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, tag2id),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./models",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model("./models/final")
    tokenizer.save_pretrained("./models/final")

if __name__ == "__main__":
    main()
