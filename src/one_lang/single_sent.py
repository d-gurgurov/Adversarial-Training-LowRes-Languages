import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from transformers import BertTokenizer
from datasets import load_dataset, Dataset
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np
import random
import argparse
import os

from sent_bert import SentimentBERT

parser = argparse.ArgumentParser(description="Train a BERT model for sentiment analysis.")
parser.add_argument('--language', type=str, default='maltese', help='Language of the dataset')
parser.add_argument('--seed', type=int, default=43, help='Random seed value')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')

args = parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# decrease precision slightly 
torch.set_float32_matmul_precision("high")

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

def load_and_prepare_data(language):
    dataset = load_dataset(f"DGurgurov/{args.language}_sa")

    df = pd.DataFrame(dataset['train'])

    combined_train_df = pd.concat([df], ignore_index=True)

    combined_val_df = pd.concat([pd.DataFrame(dataset['validation'])], ignore_index=True)
    
    combined_test_df = pd.concat([pd.DataFrame(dataset['test'])], ignore_index=True)
    
    combined_train_dataset = Dataset.from_pandas(combined_train_df)
    combined_val_dataset = Dataset.from_pandas(combined_val_df)
    combined_test_dataset = Dataset.from_pandas(combined_test_df)

    return combined_train_dataset, combined_val_dataset, combined_test_dataset

combined_train_dataset, combined_val_dataset, combined_test_dataset = load_and_prepare_data(args.language)

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_train_dataset = combined_train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = combined_val_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = combined_test_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

def create_data_loader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_data_loader(tokenized_train_dataset)
val_loader = create_data_loader(tokenized_val_dataset, batch_size=64)
test_loader = create_data_loader(tokenized_test_dataset, batch_size=64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SentimentBERT(num_classes=2)
model.to(device)
torch.compile(model)

optimizer_params = [
            {
                "params": model.bert.encoder.parameters(),
                "lr": 1e-5 
            },
            {
                "params": model.classifier.parameters(),
                "lr": 1e-5 
            }
]

optimizer = optim.AdamW(optimizer_params,  betas=(0.98, 0.95), eps=1e-08)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * args.num_epochs)
criterion = nn.CrossEntropyLoss()

def train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs=10, model_save_path='best_single_lang_model.pth'):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        val_loss, val_f1 = evaluate_model(model, val_loader)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Val Sentiment F1: {val_f1:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Saved best model with validation loss: {val_loss:.4f}')

def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return avg_loss, f1

# Train the model
train_model(model, optimizer, scheduler, criterion, train_loader, val_loader, num_epochs=args.num_epochs)

model_save_path='best_single_lang_model.pth'
model.load_state_dict(torch.load(model_save_path))

# Evaluate on test set
test_loss, test_f1 = evaluate_model(model, test_loader)
print(f'Test Loss: {test_loss:.4f}, '
      f'Test Sentiment F1: {test_f1:.4f}')

# Save the test results to a file
results_file = 'results/single_sent_results.txt'
os.makedirs(os.path.dirname(results_file), exist_ok=True)
with open(results_file, 'a') as f:
    f.write(f'Language: {args.language}, Seed: {args.seed}, Num Epochs: {args.num_epochs}\n')
    f.write(f'Test Loss: {test_loss:.4f}, Test Sentiment F1: {test_f1:.4f}\n\n')
    f.write(f' ')
