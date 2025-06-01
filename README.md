# Linformer – A Practical Way to Scale Transformers Efficiently

This repository contains an implementation of the Linformer model, a variant of the Transformer architecture that reduces the complexity of self-attention from O(n²) to O(n), making it more efficient for longer sequences.

All code is provided in a single Jupyter Notebook and includes training, inference, and ablation experiments comparing Linformer with standard Transformer models.

---

## 📓 Project Contents

- ✅ Linformer model implementation  
- ✅ Training pipeline and inference code  
- ✅ Loss curves and performance visualizations  
- ✅ Ablation studies: training & inference time comparison  

---

## Model Architecture
![Architecture](Assets/Architecture.png)

- Linformer replaces the full self-attention with projected key and value matrices, reducing the attention complexity from $O(n^2)$ to $O(n)$.
- The model architecture remains compatible with the standard Transformer pipeline, allowing seamless integration into existing frameworks while enabling faster training and inference on long sequences.

---

## How it works?

<p align="center">
  <img src="Assets/model_equation.png" alt="Model Equation" width="500"/>
</p>

- Linformer introduces a linear self-attention mechanism by projecting the key and value matrices using low-rank linear projections Eᵢ and Fᵢ, reducing their shape from n × d to k × d.
- This reduces the attention complexity from O(n²) to O(nk), making it much more efficient when k ≪ n.

---

## 🛠️ Training Pipeline

The training process for the Linformer model is structured to be efficient, modular, and reproducible. Below is a high-level overview of the key stages in the pipeline:

1. **Library Imports**  
   Essential Python libraries and deep learning frameworks (e.g., PyTorch, NumPy) are imported to support model development and training.

2. **Tokenization**  
   The dataset is tokenized using the [Tiktoken](https://github.com/openai/tiktoken) tokenizer configured for the GPT-2 vocabulary, ensuring compatibility with Transformer-based architectures.

3. **Data Splitting**  
   The tokenized dataset is split into training and validation sets to monitor generalization performance during training.

4. **Model Configuration**  
   A configuration object is defined, specifying hyperparameters such as model depth, number of heads, sequence length, hidden dimensions, dropout rates, etc.

5. **Model Components**  
   Core building blocks of the Linformer architecture are implemented modularly, including:
   - Token and positional embeddings  
   - Linear self-attention mechanism with low-rank projections  
   - Feedforward layers  
   - Transformer block structure (LayerNorm, residual connections, etc.)

6. **Model Definition**  
   The full Linformer model is instantiated by stacking the necessary number of Transformer blocks and adding a final linear head for prediction.

7. **Learning Rate Scheduler**  
   A learning rate scheduler is used to stabilize and accelerate training:
   - **Warm-up phase:** Gradually increases the learning rate for the first 5% of total training iterations  
   - **Cosine decay:** Smoothly decays the learning rate following a cosine schedule for the remaining steps

This modular and well-structured pipeline ensures clarity, ease of experimentation, and efficient training on long sequences.
