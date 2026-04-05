# Audio Models - Knowledge Base Index

A comprehensive collection of research papers and models focused on speech processing, including automatic speech recognition (ASR), speech translation, and related audio technologies.

## Overview

This folder contains documentation and implementations of state-of-the-art audio models covering:
- **Efficient Speech Recognition**: Fast inference with competitive accuracy
- **Multilingual Models**: Support for 15-25+ languages
- **Speech Translation**: End-to-end speech-to-text translation
- **Sequence Transduction**: Advanced attention and duration prediction mechanisms

---

## Papers & Models

### 1. **Fast-Conformer-Efficient-Speech-Recognition** (2305.05084)

**Title**: Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition

**Authors**: NVIDIA Research (Dima Rekesh, Nithin Rao Koluguri, Boris Ginsburg, et al.)

**Summary**:
Introduces Fast Conformer, a redesigned Conformer architecture optimized for efficient training and inference. Key improvements include:
- **2.8× faster** inference than original Conformer
- 8× downsampling schema (vs. 4× in standard Conformer)
- Supports scaling to **1 Billion parameters** without architectural changes
- Limited context attention for long-form audio (up to 11 hours transcription)
- State-of-the-art accuracy on ASR benchmarks (LibriSpeech, MLS, Common Voice, WSJ)

**Applications**:
- Automatic Speech Recognition (ASR)
- Speech Translation (ST) - outperforms Conformer with 1.66× speedup
- Spoken Language Understanding (SLU) - Intent classification and slot filling
- Long-form audio transcription with global attention tokens

**Performance Highlights**:
- LibriSpeech test-other: 4.99% WER (RNNT)
- Scales to 1.1B parameters with improved accuracy and noise robustness
- Trained on 25k hours of English ASR data

---

### 2. **Fast-Conformer-Linearly-Scalable-v6** (2305.05084v6)

**Title**: Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition (Version 6)

**Authors**: NVIDIA Research

**Summary**:
An updated version of the Fast Conformer paper with refined experiments and expanded evaluation. Includes:
- Detailed ablation studies on downsampling components
- Extended evaluation on multilingual benchmarks
- Improved long-form audio handling with limited context attention
- Performance comparison with SqueezeFormer and EfficientConformer variants

**Key Differences from v1**:
- More comprehensive scaling experiments (L, XL, XXL sizes)
- Additional dataset: ASR Set++ with 40,000 hours
- Expanded results on multiple long-form audio benchmarks
- Detailed noise robustness analysis

---

### 3. **CoVoST-2-Multilingual-Speech-Translation** (2007.10310)

**Title**: CoVoST 2 and Massively Multilingual Speech-to-Text Translation

**Authors**: Facebook AI Research (Changhan Wang, Anne Wu, Juan Pino)

**Summary**:
Introduces CoVoST 2, the largest open-source speech-to-text translation corpus to date. Features:
- **21 source languages → English** translations
- **English → 15 target languages** translations
- **2,880 hours** of speech data (extended from 700 hours in CoVoST v1)
- **78K speakers** across diverse backgrounds
- Extensive baseline models for ASR, MT, and ST tasks

**Key Features**:
- Languages covered: French, German, Spanish, Catalan, Italian, Russian, Chinese, Portuguese, Persian, Estonian, Mongolian, Dutch, Turkish, Arabic, Swedish, Latvian, Slovenian, Tamil, Japanese, Indonesian, Welsh
- Comprehensive quality control using language model perplexity and LASER scores
- Both monolingual and multilingual baseline experiments
- CC0 open license for community research

**Applications**:
- Massively multilingual ST research
- Low-resource language pair translation
- Many-to-one and one-to-many translation studies
- Self-supervised and semi-supervised learning approaches

---

### 4. **Token-Duration-Transducer-TDT** (2304.06795)

**Title**: Efficient Sequence Transduction by Jointly Predicting Tokens and Durations

**Authors**: NVIDIA Research (Hainan Xu, Somshubra Majumdar, Boris Ginsburg, et al.) & Carnegie Mellon University

**Summary**:
Introduces Token-and-Duration Transducer (TDT), an enhanced RNN-Transducer architecture that jointly predicts both tokens AND durations (number of frames per token). Benefits include:
- **2.82× faster inference** on Speech Recognition vs. conventional Transducers
- **2.27× faster inference** on Speech Translation
- Better accuracy across all tested tasks
- Skips input frames guided by predicted durations (more efficient than frame-by-frame processing)

**Performance Improvements**:
- **Speech Recognition**: Better accuracy + 2.82× faster inference
- **Speech Translation**: +1 BLEU absolute gain on MUST-C dataset + 2.27× speedup
- **Intent Classification & Slot Filling**: +1% intent accuracy + 1.28× faster

**Technical Innovation**:
- Joint network with two outputs: token distribution and duration distribution
- Independently normalized predictions for robustness
- Enables variable-length frame skipping during inference

---

### 5. **Canary-Parakeet-Multilingual-ASR** (NVIDIA)

**Title**: Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST

**Authors**: NVIDIA Research (Monica Sekoyan, Nithin Rao Koluguri, Nune Tadevosyan, et al.)

**Summary**:
Introduces two production-ready multilingual models for automatic speech recognition and speech-to-text translation:

**Canary-1B-v2**:
- **1 Billion parameter** model
- FastConformer encoder + Transformer decoder
- Supports **25 European languages**
- Trained on **1.7M hours** of speech data (Granary + NeMo ASR Set 3.0)
- **10× faster** than Whisper-large-v3 while maintaining competitive accuracy
- Two-stage pre-training and fine-tuning with dynamic data balancing

**Parakeet-TDT-0.6B-v3**:
- **600M parameter** model (lightweight alternative)
- Successor to v2
- Multilingual ASR across 25 languages
- Uses Token-Duration-Transducer architecture

**Key Features**:
- Non-speech audio augmentation to reduce hallucinations
- NeMo Forced Aligner (NFA) for timestamp generation
- Outperforms Whisper-large-v3 on English ASR
- Competitive performance vs. Seamless-M4T-v2-large despite smaller size
- Optimized for both cloud and edge deployment

**Performance**:
- English ASR: Outperforms Whisper-large-v3
- 10× faster inference than Whisper-large-v3
- Competitive multilingual ASR across 25 languages
- Reliable segment-level timestamps via NFA

---

## Model Comparison Matrix

| Model | Size | Languages | Speed | Accuracy | Best For |
|-------|------|-----------|-------|----------|----------|
| **Fast Conformer** | 120M-1.1B | English + DE | 2.8× | SOTA | General ASR, Long-form |
| **CoVoST 2** | Baseline | 21→En, En→15 | Varies | Baseline | Research, Low-resource |
| **Token-Duration-Transducer** | Varies | Multiple | 2.27-2.82× | Improved | ST, SLU, Efficiency |
| **Canary-1B-v2** | 1B | 25 European | 10× | SOTA | Production multilingual ASR |
| **Parakeet-TDT-0.6B** | 600M | 25 European | Fast | Good | Edge/mobile deployment |

---

## Implementation Details

All models are open-sourced through:
- **NVIDIA NeMo Toolkit**: https://github.com/NVIDIA/NeMo
- **FairSeq**: https://github.com/pytorch/fairseq (CoVoST 2 baselines)
- **Hugging Face**: Community implementations and model cards

## Key Takeaways

1. **Efficiency Gains**: Modern speech models achieve 2-10× speedups over baselines without sacrificing accuracy
2. **Multilingual Capability**: Supporting 15-25 languages is now standard for production models
3. **Attention Mechanisms**: Limited context + global tokens enable long-form audio processing (11+ hours)
4. **Joint Prediction**: Token-duration joint modeling improves both speed and accuracy
5. **Scale Benefits**: Billion-parameter models show continued improvements with larger datasets (1.7M+ hours)

---

## References & Related Work

- **Conformer**: Original Conformer architecture (Gulati et al., 2020)
- **LongFormer**: Long-document attention mechanisms (Beltagy et al., 2020)
- **Wav2Vec 2.0**: Self-supervised speech representation learning
- **MuST-C**: Multilingual speech translation corpus (Di Gangi et al., 2019)
- **Whisper**: OpenAI's multilingual speech recognition model

---

**Last Updated**: October 24, 2025
