import json
import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from winner import training

# Logger setup
logger = logging.getLogger("winner")
logger_stream_handler = logging.StreamHandler()
logger_stream_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(logger_stream_handler)
logger.setLevel(logging.DEBUG)

trainedModel = None
tokenizer = None

def setup_model():
    model_name = os.getenv("MODEL_NAME", "bert-base-cased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return tokenizer, model

def read_iob_data(data_string):
    sentences, tags = [], []
    current_sent, current_tags = [], []
    
    for line in data_string.split('\n'):
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

def train(trainData: str):
    global trainedModel, tokenizer
    logger.info("In train, processing training data")
    
    sentences, tags = read_iob_data(trainData)
    unique_tags = sorted(list(set(tag for tag_list in tags for tag in tag_list)))
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    
    tokenizer, model = setup_model()
    
    train_encodings = tokenizer(sentences, is_split_into_words=True, padding=True, truncation=True)
    train_labels = [[tag2id[t] for t in doc] for doc in tags]
    
    training_args = TrainingArguments(
        output_dir="./models",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=list(zip(train_encodings, train_labels))
    )
    
    trainer.train()
    trainer.save_model("models/model-best")
    tokenizer.save_pretrained("models/model-best")
    
    trainedModel = model
    logger.info("Training finished")
    return True

def evaluate(evalData: training.EvalData) -> (training.EvalOutput, bool):
    global trainedModel, tokenizer
    logger.info("In evaluate, received evalData: " + evalData.data)
    
    if trainedModel is None:
        tokenizer = AutoTokenizer.from_pretrained("models/model-best")
        trainedModel = AutoModelForTokenClassification.from_pretrained("models/model-best")
        trainedModel.eval()
    
    sentences = [word for word in evalData.data.split() if word.strip()]
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = trainedModel(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
    
    output = []
    for token, pred in zip(sentences, predictions[0]):
        label = trainedModel.config.id2label[pred.item()]
        output.append(f"{token}\t{label}")
    
    logger.info(output)
    evalOutput = training.EvalOutput("\n".join(output))
    return (evalOutput, True)

def main():
    logger.info("In main")
    trainAddress = os.getenv("training_address", "localhost:7707")
    trainer = training.Train(train, evaluate, trainAddress)
    success = trainer.Begin()
    logger.info(f"Training finished with status: {success}")

if __name__ == "__main__":
    main()
