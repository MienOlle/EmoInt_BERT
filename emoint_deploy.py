def extract_all_lexicons(text):
    vad_feats = extract_vad(text, nrc_vad_lexicon)
    lex_feats = extract_lex(text, nrc_lexicon)
    hash_feats = extract_hash_emo(text, hash_emo_lex)
    
    combined_feats = np.concatenate([vad_feats, lex_feats, hash_feats])
    return combined_feats

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
    lexicon_feats = torch.tensor([extract_all_lexicons(clean)], dtype=torch.float).to(device)

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