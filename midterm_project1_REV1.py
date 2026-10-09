import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. SIMPLE TOKENIZER & VOCABULARY
# ==========================================
class Vocabulary:
    def __init__(self, max_size=10000, unk_token="<UNK>", pad_token="<PAD>"):
        self.max_size = max_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}

    def build_vocab(self, texts):
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        most_common = word_counts.most_common(self.max_size - 2)
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def tokenize(self, text):
        if not isinstance(text, str):
            return []
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return text.split()

    def encode(self, text, max_len=256):
        tokens = self.tokenize(text)
        indices = [self.word2idx.get(tok, self.word2idx[self.unk_token]) for tok in tokens[:max_len]]
        if len(indices) < max_len:
            indices += [self.word2idx[self.pad_token]] * (max_len - len(indices))
        return indices

# ==========================================
# 2. DATASET CLASS FOR LLM FINGERPRINTING
# ==========================================
class LLMDataset(Dataset):
    def __init__(self, df, vocab, input_mode="output_only", max_len=256):
        """
        input_mode: 'input_only', 'output_only', or 'input_output'
        """
        self.labels = df['label_idx'].values
        self.vocab = vocab
        self.max_len = max_len
        
        if input_mode == "input_only":
            self.texts = df['LLM_Input'].astype(str).tolist()
        elif input_mode == "output_only":
            self.texts = df['LLM_output'].astype(str).tolist()
        elif input_mode == "input_output":
            self.texts = (df['LLM_Input'].astype(str) + " " + df['LLM_output'].astype(str)).tolist()
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text_encoded = self.vocab.encode(self.texts[idx], max_len=self.max_len)
        return torch.tensor(text_encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# ==========================================
# 3. TEXT CNN MODEL
# ==========================================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, filter_sizes=[3, 4, 5], num_filters=100, dropout=0.3):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, seq_len, embed_dim]
        
        conv_results = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_results = [torch.max_pool1d(c, c.size(2)).squeeze(2) for c in conv_results]
        
        out = torch.cat(pooled_results, 1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

# ==========================================
# 4. RECURRENT NEURAL NETWORK (BiLSTM)
# ==========================================
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, num_layers=2, dropout=0.3, bidirectional=True):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        dirs = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_dim * dirs, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        out, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            hidden_out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden_out = hidden[-1, :, :]
            
        hidden_out = self.dropout(hidden_out)
        logits = self.fc(hidden_out)
        return logits

# ==========================================
# 5. TRAINING AND EVALUATION LOOPS
# ==========================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(y_batch)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
        
    return total_loss / total, correct / total

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            
            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader.dataset)
    acc = (np.array(all_preds) == np.array(all_targets)).mean()
    return avg_loss, acc, all_preds, all_targets

# ==========================================
# 6. FEATURE ANALYSIS (RQ4 HELPER)
# ==========================================
def extract_linguistic_features(df):
    """Extract structural & stylistic signals for RQ4 analysis"""
    features = pd.DataFrame()
    features['len_chars'] = df['LLM_output'].apply(lambda x: len(str(x)))
    features['len_words'] = df['LLM_output'].apply(lambda x: len(str(x).split()))
    features['markdown_count'] = df['LLM_output'].apply(lambda x: str(x).count('```') + str(x).count('**'))
    features['bullet_count'] = df['LLM_output'].apply(lambda x: str(x).count('- ') + str(x).count('* '))
    features['ttr'] = df['LLM_output'].apply(
        lambda x: len(set(str(x).lower().split())) / (len(str(x).split()) + 1e-5)
    )
    return features

# ==========================================
# 7. EXPERIMENT RUNNER (PIPELINE EXECUTION)
# ==========================================
def run_experiment(model_type, train_df, test_df, vocab, num_classes, label_names, input_mode="output_only", epochs=5, device="cpu"):
    train_dataset = LLMDataset(train_df, vocab, input_mode=input_mode)
    test_dataset = LLMDataset(test_df, vocab, input_mode=input_mode)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    vocab_size = len(vocab.word2idx)
    embed_dim = 128

    if model_type == "CNN":
        model = TextCNN(vocab_size=vocab_size, embed_dim=embed_dim, num_classes=num_classes).to(device)
    elif model_type == "RNN":
        model = TextRNN(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=128, num_classes=num_classes).to(device)
    else:
        raise ValueError("Invalid model_type")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\n--- Training {model_type} (Input Mode: '{input_mode}') ---")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    _, _, final_preds, final_targets = evaluate(model, test_loader, criterion, device)
    print(f"\nClassification Report ({model_type} - {input_mode}):")
    print(classification_report(final_targets, final_preds, target_names=label_names, zero_division=0))
    print(f"Confusion Matrix ({model_type} - {input_mode}):")
    print(confusion_matrix(final_targets, final_preds))
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- STEP 1: LOAD YOUR DATASET ---
    try:
        df = pd.read_csv("llm_dataset.csv")
        print(f"Loaded dataset 'llm_dataset.csv' with {len(df)} rows.")
    except FileNotFoundError:
        print("\nDataset file 'llm_dataset.csv' not found! Generating dummy dataset for testing pipeline...")
        data = {
            'LLM_name': ['OpenAI (GPT)', 'Anthropic (Claude)', 'Meta (Llama)'] * 200,
            'LLM_Input': ['Explain gravity simply.'] * 300 + ['Write a python function for quicksort.'] * 300,
            'LLM_output': [
                'Gravity is a fundamental force pulling objects together in space.' if i % 3 == 0 
                else 'Here is a comprehensive breakdown of how gravity operates according to general relativity.' if i % 3 == 1 
                else 'Think of gravity like a heavy ball placed on a stretched rubber sheet or trampoline.' 
                for i in range(600)
            ],
            'task_type': ['general_qa'] * 300 + ['coding'] * 300
        }
        df = pd.DataFrame(data)

    # --- STEP 2: PREPROCESS LABELS AND VOCABULARY ---
    label_names = sorted(df['LLM_name'].unique().tolist())
    label2idx = {name: idx for idx, name in enumerate(label_names)}
    df['label_idx'] = df['LLM_name'].map(label2idx)
    num_classes = len(label_names)

    vocab = Vocabulary(max_size=10000)
    all_texts = df['LLM_Input'].astype(str).tolist() + df['LLM_output'].astype(str).tolist()
    vocab.build_vocab(all_texts)
    print(f"Built vocabulary with {len(vocab.word2idx)} unique words for {num_classes} classes: {label_names}")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_idx'])

    # --- STEP 3: RQ1 — Identify LLMs using Output Only ---
    print("\n==================== EXECUTING RQ1 ====================")
    cnn_model_rq1 = run_experiment("CNN", train_df, test_df, vocab, num_classes, label_names, input_mode="output_only", epochs=5, device=device)
    rnn_model_rq1 = run_experiment("RNN", train_df, test_df, vocab, num_classes, label_names, input_mode="output_only", epochs=5, device=device)

    # --- STEP 4: RQ2 — Role of User Prompt ---
    print("\n==================== EXECUTING RQ2 ====================")
    run_experiment("CNN", train_df, test_df, vocab, num_classes, label_names, input_mode="input_only", epochs=5, device=device)
    run_experiment("CNN", train_df, test_df, vocab, num_classes, label_names, input_mode="input_output", epochs=5, device=device)

    # --- STEP 5: RQ3 (EXTRA CREDIT) — Cross-Domain Generalization ---
    print("\n==================== EXECUTING RQ3 (CROSS-DOMAIN) ====================")
    if 'task_type' in df.columns and len(df['task_type'].unique()) > 1:
        train_rq3 = df[df['task_type'] != 'coding']
        test_rq3 = df[df['task_type'] == 'coding']
        if len(train_rq3) > 10 and len(test_rq3) > 10:
            print(f"Training on Non-Coding ({len(train_rq3)} samples), Testing on Coding ({len(test_rq3)} samples)...")
            run_experiment("CNN", train_rq3, test_rq3, vocab, num_classes, label_names, input_mode="output_only", epochs=5, device=device)
        else:
            print("Insufficient data split for RQ3 domain evaluation.")
    else:
        print("Skipping RQ3: Single task_type detected in dataset.")

    # --- STEP 6: RQ4 (EXTRA CREDIT) — Stylistic Analysis ---
    print("\n==================== EXECUTING RQ4 ANALYSIS ====================")
    features_df = extract_linguistic_features(df)
    features_df['LLM_name'] = df['LLM_name']
    print("\nMean Linguistic Metrics per LLM Family:")
    print(features_df.groupby('LLM_name').mean())