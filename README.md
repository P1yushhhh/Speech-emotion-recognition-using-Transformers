# Speech Emotion Recognition using Wav2Vec2 Transformer

A deep learning project that leverages the power of Wav2Vec2 transformer architecture to recognize emotions from speech audio. This model achieves an impressive **95% accuracy** on the RAVDESS dataset for English language emotion classification.

## Overview

This project implements a speech emotion recognition system that can classify various emotional states from audio recordings. The system uses Wav2Vec2 2.0 as the feature extractor to capture rich contextual embeddings from speech signals, followed by a custom neural network for emotion classification.

## Dataset

The model is trained and evaluated on the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset, which contains:
- High-quality audio recordings
- Multiple emotional categories including neutral, calm, happy, sad, angry, fearful, disgust, and surprised
- English language speech samples
- Balanced representation across different emotions

## Architecture

### Feature Extraction
- **Preprocessing**: Audio signals undergo noise removal and normalization
- **Wav2Vec2 2.0**: Extracts deep, contextual embeddings from raw audio signals
- **Input Processing**: Raw audio recordings (.wav) resampled to 16 kHz from RAVDESS dataset
- **Embedding Dimension**: 768-dimensional feature vectors at input
- **Pooling**: Mean pooling applied to obtain fixed-size vectors

### Model Architecture
The proposed neural network consists of:
```
Linear(768 → 256) → ReLU → Dropout(0.3) → Linear(256 → 8) → Softmax
```

**Training Configuration:**
- Optimizer: AdamW with weight decay of 0.01
- Output: 8 emotion classes using softmax activation
- Loss Function: Cross-entropy loss for multi-class classification

## Results

The model demonstrates excellent performance on emotion recognition:

### Overall Performance
- **Accuracy**: 95%
- **F1-Score (Macro)**: 0.9492
- **F1-Score (Micro)**: 0.9531
- **F1-Score (Weighted)**: 0.9530

### Per-Class Performance

| Emotion   | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Neutral   | 0.79      | 1.00   | 0.88     | 38      |
| Calm      | 0.99      | 0.95   | 0.97     | 76      |
| Happy     | 1.00      | 0.84   | 0.92     | 77      |
| Sad       | 0.95      | 0.96   | 0.95     | 77      |
| Angry     | 1.00      | 0.97   | 0.99     | 77      |
| Fearful   | 1.00      | 0.92   | 0.96     | 77      |
| Disgust   | 1.00      | 1.00   | 1.00     | 77      |
| Surprised | 0.85      | 1.00   | 0.92     | 77      |

## Key Features

- **State-of-the-art Architecture**: Utilizes Wav2Vec2 2.0 transformer for robust feature extraction
- **High Accuracy**: Achieves 95% accuracy on emotion classification
- **Multi-emotion Support**: Recognizes 8 different emotional states
- **Efficient Processing**: Optimized for real-time emotion recognition
- **Robust Performance**: Consistent results across different emotional categories

## Technical Highlights

- **Deep Contextual Understanding**: Wav2Vec2 captures rich semantic information from speech patterns
- **End-to-end Learning**: Direct processing from raw audio to emotion labels
- **Balanced Classification**: Strong performance across all emotion categories
- **Regularization**: Dropout layers prevent overfitting and improve generalization

## Applications

This speech emotion recognition system can be applied in various domains:
- Customer service sentiment analysis
- Mental health monitoring
- Human-computer interaction
- Educational technology
- Entertainment and gaming

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- librosa for audio processing
- scikit-learn for evaluation metrics

The model demonstrates the effectiveness of combining transformer-based feature extraction with traditional neural network architectures for emotion recognition tasks, achieving state-of-the-art performance on the RAVDESS dataset.
