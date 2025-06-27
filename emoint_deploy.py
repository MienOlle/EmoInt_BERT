import numpy as np
import pandas as pd
import torch
import re
import emoji
import contractions
from collections import defaultdict
import joblib
from transformers import BertTokenizer, BertModel
import torch.nn as nn
from torch.nn import functional as F
import streamlit as st

def load_lex(filepath):
    lexicon = defaultdict(dict)
    with open(filepath, 'r') as file:
        for line in file:
            word, emotion, value = line.strip().split('\t')
            if int(value) == 1:
                lexicon[word][emotion] = 1
    return lexicon

def load_nrc_vad(filepath):
    vad_lex = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        for line in f:
            word, val, aro, dom = line.strip().split('\t')
            vad_lex[word] = {
                'valence': float(val),
                'arousal': float(aro),
                'dominance': float(dom)
            }
    return vad_lex

def load_nrc_hash_emo(filepath):
    lexicon = defaultdict(dict)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            emotion, word, score = line.strip().split('\t')
            lexicon[word][emotion] = float(score)
    return lexicon

def tokenize_texts(texts, max_len=128):
    return tokenizer(
        texts.tolist(),
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )

def convert_emojis(text):
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = re.sub(r':([a-zA-Z_]+):', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_text(text):
    text = text.lower()
    text = contractions.fix(text)
    text = convert_emojis(text)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r"[^a-zA-Z\s.,!?']", '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_lex(text, lexicon):
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
              'sadness', 'surprise', 'trust', 'positive', 'negative']
    counts = dict.fromkeys(emotions, 0)

    for word in text.split():
        if word in lexicon:
            for emo in lexicon[word]:
                counts[emo] += 1
    return [counts[emo] for emo in emotions]

def extract_vad(text, lexicon):
    valence = []
    arousal = []
    dominance = []

    for word in text.split():
        if word in lexicon:
            valence.append(lexicon[word]['valence'])
            arousal.append(lexicon[word]['arousal'])
            dominance.append(lexicon[word]['dominance'])

    # If no word matched, return zeros
    if not valence:
        return [0.0, 0.0, 0.0]

    # Otherwise, return means
    return [
        np.mean(valence),
        np.mean(arousal),
        np.mean(dominance)
    ]

def extract_hash_emo(text, lexicon):
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy',
                'sadness', 'surprise', 'trust']
    scores = {emo: [] for emo in emotions}

    for word in text.split():
        if word in lexicon:
            for emo, value in lexicon[word].items():
                scores[emo].append(value)

    return [np.mean(scores[emo]) if scores[emo] else 0.0 for emo in emotions]

def extract_all_lexicons(text):
    vad_feats = extract_vad(text, nrc_vad_lexicon)
    vad_feats = scaler_vad.transform([vad_feats])
    lex_feats = extract_lex(text, nrc_lexicon)
    lex_feats = scaler_lex.transform([lex_feats])
    hash_feats = extract_hash_emo(text, hash_emo_lex)
    hash_feats = scaler_hash.transform([hash_feats])

    combined_feats = np.concatenate([vad_feats, lex_feats, hash_feats], axis = 1)
    return combined_feats

class EmotionMultiTaskModel(nn.Module):
    def __init__(self, num_emotions=4, lex_dim=21):
        super(EmotionMultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)

        # Shared representation
        hidden_size = self.bert.config.hidden_size
        self.shared_layer = nn.Linear(hidden_size + lex_dim, hidden_size)

        # Task-specific layers
        self.classifier = nn.Linear(hidden_size, num_emotions)  # Multi-label classification
        self.regressor = nn.Linear(hidden_size, num_emotions)   # Multi-output regression

    def forward(self, input_ids, attention_mask, lexicon_feats):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Concatenate with lexicon features
        combined = torch.cat((pooled_output, lexicon_feats), dim=1)

        # Shared representation
        shared_repr = F.relu(self.shared_layer(combined))
        shared_repr = self.dropout(shared_repr)

        # Task-specific outputs
        cls_logits = self.classifier(shared_repr)  # For binary classification of each emotion
        reg_output = self.regressor(shared_repr)   # For regression of each emotion's intensity

        # Apply sigmoid to classification logits
        cls_probs = torch.sigmoid(cls_logits)

        # Scale regression outputs to [0,1]
        reg_output = (torch.tanh(reg_output) + 1) / 2

        return cls_probs, reg_output
    
nrc_lexicon = load_lex("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")
nrc_vad_lexicon = load_nrc_vad("NRC-VAD-Lexicon-v2.1.txt")
hash_emo_lex = load_nrc_hash_emo("NRC-Hashtag-Emotion-Lexicon-v0.2.txt")

scaler_lex = joblib.load("lex_scaler.pkl")
scaler_vad = joblib.load("vad_scaler.pkl")
scaler_hash = joblib.load("hash_scaler.pkl")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
emotion_cols = ["joy", "sadness", "anger", "fear"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_emotions = len(emotion_cols)
lex_dim = 21
model = EmotionMultiTaskModel(num_emotions=num_emotions, lex_dim=lex_dim).to(device)
model.load_state_dict(torch.load("best_multitask_multilabel_model.pth", map_location=device))

def predict_emotions(text, model, tokenizer, threshold=0.3):
    model.eval()

    # Clean and tokenize the text
    clean = clean_text(text)
    tokens = tokenizer(
        clean,
        padding='max_length',
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

    # Create lexicon features
    lexicon_feats = torch.tensor(extract_all_lexicons(clean), dtype=torch.float).to(device)

    # Move inputs to device
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)

    # Get predictions
    with torch.no_grad():
        cls_probs, intensities = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            lexicon_feats=lexicon_feats
        )

        # Convert to numpy
        cls_probs = cls_probs.cpu().numpy()[0]
        intensities = intensities.cpu().numpy()[0]

        # Apply threshold to classification probabilities
        # detected_emotions = cls_probs > threshold
        detected_emotions = np.zeros_like(cls_probs, dtype=bool)
        detected_emotions[cls_probs.argmax()] = True

    # Prepare results
    results = {}
    for i, emotion in enumerate(emotion_cols):
        results[emotion] = {
            "probability": float(cls_probs[i]),
            "detected": bool(detected_emotions[i]),
            "intensity": float(intensities[i]) if detected_emotions[i] else 0.0
        }

    return results


text = "I'm angry, but I think I can tolerate their behavior."
result = predict_emotions(text, model, tokenizer)
print(f"Text: {text}")
print("Detected emotions:")

# Sort emotions by intensity
emotions_sorted = sorted(
    [(emotion, details) for emotion, details in result.items() if details["detected"]],
    key=lambda x: x[1]["intensity"],
    reverse=True
)

if emotions_sorted:
    for emotion, details in emotions_sorted:
        print(f"  {emotion}: intensity={details['intensity']:.2f}, probability={details['probability']:.2f}")
else:
    print("  No emotions detected")
print("---")