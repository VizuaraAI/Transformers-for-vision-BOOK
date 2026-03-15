# Transformers for Vision

Code repository for the **Transformers for Vision** book.

> **Note:** Currently only Chapter 2 code is available. Code for all other chapters will be uploaded in the future.

---

## Table of Contents

### Chapter 1: Introduction – From CNNs to the Need for Transformers
- Evolution of computer vision: from handcrafted features to deep learning
- CNNs and their inductive biases (locality, translation invariance)
- Strengths and limitations of CNNs
- Why transformers became attractive for vision (parallelization, global context)
- The spark from NLP: attention as a universal mechanism
- Case study: CNN vs attention on a sample task
- Hands-on: implementing a small CNN and comparing with a simple attention model

### Chapter 2: The Transformer Architecture – A Deep Dive
- 2.1 Introduction to Large Language Models
- 2.2 Anatomy of the Transformer Block
- 2.3 Tokenization
- 2.4 Byte Pair Encoding
- 2.5 Word Embedding
- 2.6 Transformer Block
- 2.7 The Need for Attention Mechanism
- 2.8 Self Attention Mechanism
- 2.9 Understanding the Input Embedding Matrix
- 2.10 From Embeddings to Queries, Keys & Values
- 2.11 A Quick Note on Matrix Multiplication
- 2.12 Why Scale Attention Scores?
- 2.13 Causal & Masked Attention
- 2.14 Causal Attention with Dropouts
- 2.15 Summary of Self-Attention
- 2.16 Intuition of Multi-Head Attention
- 2.17 Layer Normalization
- 2.18 FeedForward Network
- 2.19 Shortcut Connections
- 2.20 Why Transformers Scale Better Than RNNs and CNNs
- 2.21 Pretraining, Fine Tuning, and Transfer Learning in Transformers
- 2.22 Limitations and Challenges of Transformers
- 2.23 Hands-On: Coding a Miniature Transformer for Sequence Classification
- 2.24 Summary

### Chapter 3: Vision Transformers (ViT)
- Adapting transformers to images: patch embeddings and flattening
- Positional encodings in vision
- Encoder-only structure for classification
- Benefits and drawbacks of ViT
- Real-world applications of ViT
- Hands-on: fine-tuning ViT for image classification

### Chapter 4: Efficient and Scalable Vision Transformers
- Why ViT is data-hungry and computationally heavy
- DeiT: data-efficient training and knowledge distillation
- Swin Transformer: hierarchical and multi-scale representation
- Comparisons: DeiT vs Swin vs ViT
- Hands-on: training DeiT and Swin on a benchmark dataset

### Chapter 5: Transformers for Detection and Segmentation
- Object detection revisited: from R-CNN to anchor-free detection
- DETR: detection as set prediction
- Bipartite matching and transformer decoders
- Mask2Former: unifying semantic, instance, and panoptic segmentation
- SAM: Segment Anything as a foundation segmentation model
- Hands-on: detection and segmentation project

### Chapter 6: Transformers for Video Understanding
- Extending attention to temporal sequences
- TimeSformer: factorized vs joint spatio-temporal attention
- VideoMAE: masked autoencoders for video
- Applications: video retrieval, surveillance, action recognition
- Hands-on: video classification with TimeSformer

### Chapter 7: Bridging Vision and Language
- Motivation for multimodality: why vision needs language
- CLIP: aligning text and images via contrastive learning
- Zero-shot transfer with CLIP
- BLIP: generative multimodal tasks (captioning, VQA)
- Hands-on: multimodal retrieval and captioning system

### Chapter 8: Few-Shot and Large Multimodal Models
- The data-efficiency problem in multimodal learning
- Flamingo: frozen LLM + visual encoder with cross-attention
- Few-shot multimodality in practice
- Instruction tuning for multimodal models
- Hands-on: few-shot multimodal task with Flamingo-inspired setup

### Chapter 9: Generative Vision Models
- From GANs to diffusion models
- Stable Diffusion architecture: text-to-image generation
- ControlNet: adding controllability with pose, depth, and edges
- Applications in art, design, healthcare, and entertainment
- Hands-on: building your own image generation pipeline

### Chapter 10: Large Multimodal LLMs and the Future of Transformers
- LLaVA: instruction-tuned multimodal conversational agents
- Connecting LLaMA with visual encoders
- Applications in education, accessibility, robotics, and creative industries
- The future: omni-models that handle every modality
- Risks and ethical considerations: bias, misuse, hallucination
- Hands-on: building a multimodal chatbot with image + text inputs
