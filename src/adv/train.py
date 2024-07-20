import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import f1_score
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from typing import List
import numpy as np
import random
import argparse
import os
import time
from collections import Counter

from adv_bert import AdversarialBERT

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--languages', type=str, nargs='+', required=True, help='List of languages for dataset')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
args = parser.parse_args()

# random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# decrease precision slightly 
torch.set_float32_matmul_precision("high")

# initializing the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# converting to DataFrame and adding language column
def add_language_column(dataset, language_label):
    df = pd.DataFrame(dataset)
    df['language'] = language_label
    return df

print("Languages used: ", args.languages)

# loading datasets
def load_and_prepare_data(languages: List[str]):
    combined_train_df = pd.DataFrame()
    combined_val_df = pd.DataFrame()
    combined_test_df = pd.DataFrame()
    
    for i, language in enumerate(languages):
        dataset = load_dataset(f"DGurgurov/{language}_sa")
        
        train_df = add_language_column(dataset['train'], i)
        val_df = add_language_column(dataset['validation'], i)
        test_df = add_language_column(dataset['test'], i)
        
        combined_train_df = pd.concat([combined_train_df, train_df], ignore_index=True)
        combined_val_df = pd.concat([combined_val_df, val_df], ignore_index=True)
        combined_test_df = pd.concat([combined_test_df, test_df], ignore_index=True)
    
    combined_train_dataset = Dataset.from_pandas(combined_train_df)
    combined_val_dataset = Dataset.from_pandas(combined_val_df)
    combined_test_dataset = Dataset.from_pandas(combined_test_df)

    return combined_train_dataset, combined_val_dataset, combined_test_dataset

combined_train_dataset, combined_val_dataset, combined_test_dataset = load_and_prepare_data(args.languages)

# tokenizing datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_train_dataset = combined_train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = combined_val_dataset.map(tokenize_function, batched=True)
tokenized_test_dataset = combined_test_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'language'])
tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'language'])
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'language'])

print("Train dataset size:", len(tokenized_train_dataset))
print("Val dataset size:", len(tokenized_val_dataset))
print("Test dataset size:", len(tokenized_test_dataset))

# creating dataloader
def create_data_loader(dataset, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = create_data_loader(tokenized_train_dataset)
val_loader = create_data_loader(tokenized_val_dataset, batch_size=64)
test_loader = create_data_loader(tokenized_test_dataset, batch_size=64)


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# instantiating the model
model = AdversarialBERT(num_classes_sentiment=2, num_classes_language=len(args.languages), adv_heads=5)
model.to(device)
torch.compile(model)


# defining optimizers and loss functions
optimizer_params = [
            {
                "params": model.bert.encoder.parameters(),
                "lr": 1e-5 
            },
            {
                "params": model.sentiment_head.parameters(),
                "lr": 1e-5 
            },
            {
                "params": model.language_head.parameters(),
                "lr": 8e-5 
            }
        ]

num_epochs = args.num_epochs
max_grad_norm = 1.0
lmbda = 1.0 #TODO: reduce lambda influence with the current learning rate

optimizer = optim.AdamW(optimizer_params, betas=(0.98, 0.95), eps=1e-08)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * num_epochs)
criterion_sentiment = nn.CrossEntropyLoss()
criterion_language = nn.CrossEntropyLoss()

def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    total_sentiment_loss = 0
    total_language_loss = 0
    all_sentiment_preds = []
    all_sentiment_labels = []
    all_language_preds = []
    all_language_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentiment_labels = batch['label'].to(device)
            language_labels = batch['language'].to(device)
            
            sentiment_logits, language_logits_list = model(input_ids, attention_mask, lmbda)
            
            loss_sentiment = criterion_sentiment(sentiment_logits, sentiment_labels)
            loss_language = model._get_mean_loss(language_logits_list, language_labels, criterion_language)
            loss = loss_sentiment + loss_language
            
            total_loss += loss.item()
            total_sentiment_loss += loss_sentiment.item()
            total_language_loss += loss_language.item()
            
            sentiment_preds = torch.argmax(sentiment_logits, dim=1).cpu().numpy()
            all_sentiment_preds.extend(sentiment_preds)
            all_sentiment_labels.extend(sentiment_labels.cpu().numpy())

            # taking the mode prediction
            language_preds_list = [torch.argmax(logits, dim=1).cpu() for logits in language_logits_list]
            language_preds_tensor = torch.stack(language_preds_list)
            language_preds, _ = torch.mode(language_preds_tensor, dim=0)
            
            all_language_preds.extend(language_preds.numpy())
            all_language_labels.extend(language_labels.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    avg_sent_loss = total_sentiment_loss / len(data_loader)
    avg_lang_loss = total_language_loss / len(data_loader)

    sentiment_f1 = f1_score(all_sentiment_labels, all_sentiment_preds, average='macro')
    language_f1 = f1_score(all_language_labels, all_language_preds, average='macro')
    
    return avg_loss, avg_sent_loss, avg_lang_loss, sentiment_f1, language_f1


best_val_loss = float('inf')  
model_save_path = 'best_model.pth'

import time

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_sentiment_loss = 0
    total_language_loss = 0
    
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        sentiment_labels = batch['label'].to(device)
        language_labels = batch['language'].to(device)
        
        # forward pass
        sentiment_logits, language_logits_list = model(input_ids, attention_mask, lmbda)
        
        # calculate losses
        loss_sentiment = criterion_sentiment(sentiment_logits, sentiment_labels)
        loss_language = model._get_mean_loss(language_logits_list, language_labels, criterion_language)
        loss = loss_sentiment + loss_language
        
        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        total_sentiment_loss += loss_sentiment.item()
        total_language_loss += loss_language.item()
    
    # average losses for the epoch
    avg_train_loss = total_loss / len(train_loader)
    avg_train_sent_loss = total_sentiment_loss / len(train_loader)
    avg_train_lang_loss = total_language_loss / len(train_loader)
    
    # validate the model after each epoch
    val_loss, avg_val_sent_loss, avg_val_lang_loss, val_sentiment_f1, val_language_f1 = evaluate_model(model, val_loader)

    # logging the epoch results
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Sent Train Loss: {avg_train_sent_loss:.4f}, '
          f'Lang Train Loss: {avg_train_lang_loss:.4f},\n '
          f'Val Loss: {val_loss:.4f}, '
          f'Sent Val Loss: {avg_val_sent_loss:.4f}, '
          f'Lang Val Loss: {avg_val_lang_loss:.4f}, '
          f'Val Sentiment F1: {val_sentiment_f1:.4f}, '
          f'Val Language F1: {val_language_f1:.4f}')
    
    # saving the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f'Saved best model with val_loss: {val_loss:.4f}')

# final testing
model.load_state_dict(torch.load(model_save_path))
test_loss, avg_test_sent_loss, avg_test_lang_loss, test_sentiment_f1, test_language_f1 = evaluate_model(model, test_loader)

print(f'Test Loss: {test_loss:.4f}, '
      f'Test Sentiment Loss: {avg_test_sent_loss:.4f}, '
      f'Test Language Loss: {avg_test_lang_loss:.4f}, '
      f'Test Sentiment F1: {test_sentiment_f1:.4f}, '
      f'Test Language F1: {test_language_f1:.4f}')

# saving the test score to a file
file_path = 'results/adv_sent_scores.txt'
os.makedirs(os.path.dirname(file_path), exist_ok=True)

with open(file_path, 'a') as f:
    overall_results = (f'Overall Test Results - '
                       f'Test Loss: {test_loss:.4f}, '
                       f'Test Sentiment Loss: {avg_test_sent_loss:.4f}, '
                       f'Test Language Loss: {avg_test_lang_loss:.4f}, '
                       f'Test Sentiment F1: {test_sentiment_f1:.4f}, '
                       f'Test Language F1: {test_language_f1:.4f}\n')
    f.write(overall_results)

# final testing on individual test subsets
def evaluate_individual_language(model, language_dataset, language_label, lmbda):
    language_loader = create_data_loader(language_dataset, batch_size=64)
    avg_loss, avg_sent_loss, avg_lang_loss, sentiment_f1, language_f1 = evaluate_model(model, language_loader)
    
    print(f'Test Results for Language {language_label}: '
          f'Test Loss: {avg_loss:.4f}, '
          f'Sentiment Test Loss: {avg_sent_loss:.4f}, '
          f'Language Test Loss: {avg_lang_loss:.4f}, '
          f'Test Sentiment F1: {sentiment_f1:.4f}, '
          f'Test Language F1: {language_f1:.4f}')
    
    individual_results = (f'Seed {args.seed} - '
                          f'Language {language_label} - '
                          f'Test Loss: {avg_loss:.4f}, '
                          f'Sentiment Test Loss: {avg_sent_loss:.4f}, '
                          f'Language Test Loss: {avg_lang_loss:.4f}, '
                          f'Test Sentiment F1: {sentiment_f1:.4f}, '
                          f'Test Language F1: {language_f1:.4f}\n')
    with open(file_path, 'a') as f:
        f.write(individual_results)

for i, language in enumerate(args.languages):
    language_test_dataset = Dataset.from_pandas(add_language_column(load_dataset(f"DGurgurov/{language}_sa")['test'], i))
    
    tokenized_language_test_dataset = language_test_dataset.map(tokenize_function, batched=True)
    tokenized_language_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label', 'language'])
    
    evaluate_individual_language(model, tokenized_language_test_dataset, language, lmbda)
