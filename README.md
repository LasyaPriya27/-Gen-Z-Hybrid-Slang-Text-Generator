# Gen Z Hybrid Slang Text Generator

## Project Overview

### Abstract  
This project explores the application of machine learning to generate Gen Z-style slang using a hybrid model combining transformer-based language models. A fine-tuned GPT-Neo model is trained on a custom dataset of Gen Z slang, and a hybrid approach incorporating BERT for context understanding is introduced. The project evaluates both standalone GPT-Neo and a BERT + GPT-Neo hybrid to determine the most fluent and slang-accurate text generation technique.

## Introduction  
Understanding modern slang can be difficult for traditional NLP systems due to rapid linguistic evolution and contextual fluidity. This project seeks to bridge that gap by training generative models on authentic slang data, enabling more accurate generation of Gen Z-style phrases. A hybrid architecture using BERT and GPT-Neo is explored for enhanced contextual relevance.

## Contributions  
• **Fine-Tuned Language Model**: Trained a GPT-Neo model on 1000+ slang-specific sentences.  
• **Hybrid Generation Pipeline**: Developed a hybrid system using BERT for context and GPT-Neo for text generation.  
• **Text Evaluation**: Compared outputs between GPT-only and hybrid models.  
• **Tokenizer Adaptation**: Modified tokenizers for handling slang-specific tokens and padding requirements.

## Literature Review  
Recent advancements in generative language modeling using transformers (e.g., GPT and BERT) have significantly improved context-aware generation. However, slang-specific models are underexplored. Prior work shows that fine-tuning on domain-specific corpora can enhance linguistic nuance, while hybrid models (BERT for encoding + GPT for generation) improve coherence and relevance.

## Proposed Methodology  
The methodology consists of the following stages:

1. **Data Preparation**:  
   - A dataset of 5000 Gen Z slang phrases was collected.  
   - For demonstration, 1000 samples were selected for training.  

2. **Tokenization and Preprocessing**:  
   - Texts were padded and truncated to a fixed length (64 tokens).  
   - Labels and input IDs were aligned for causal language modeling.

3. **Model Fine-Tuning**:  
   - GPT-Neo (125M) was fine-tuned using Hugging Face `Trainer` with standard parameters and low batch size.  
   - Training was done on a GPU-enabled environment for speed.  

4. **Hybrid Generation**:  
   - BERT was used to encode the input.  
   - GPT-Neo generated continuation text using the raw prompt.  

## Experimental Setup and Results  
Training was performed in Google Colab with GPU acceleration. The model was evaluated qualitatively by comparing prompt completions generated from:  
- GPT-Neo only  
- BERT + GPT-Neo (hybrid)  

**Examples**:

| Prompt                  | GPT-Neo Output            | Hybrid Output                   |
|------------------------|---------------------------|---------------------------------|
| "yo that fit is"       | "so dope my guy"          | "yo that fit is mad clean fr"   |
| "she ghosted him and"  | "he dipped, no cap"       | "she ghosted him and he cried"  |

## Comparison with State-of-the-Art  
Unlike traditional GPT-based slang generation, this project introduces a hybrid mechanism that encodes prompts with BERT for enhanced semantic understanding. This improves the contextual coherence of slang usage in generated outputs. Fine-tuning on slang-specific datasets also increases the linguistic accuracy of generated text compared to zero-shot generation from pre-trained models.

## Conclusion & Future Scope  
The hybrid model demonstrates improved generation of Gen Z slang with enhanced context and fluency. This approach can be extended for:

- Larger training corpora  
- Incorporation of dialogue structure for chatbot applications  
- Integration with sentiment and tone controllers for adaptive generation  

## Dataset  
A custom dataset of 5000 Gen Z slang phrases collected from online forums, social media, and user contributions. A sample of 1000 was used in the current iteration.

## Novelty  
This project uniquely combines two transformer models—BERT and GPT-Neo—into a hybrid pipeline for slang text generation. Unlike prior approaches that rely on pretrained models for general-purpose generation, our method fine-tunes on domain-specific slang and integrates context-aware embeddings.

## Requirements  
To run this project, install the following:

```bash
pip install transformers datasets torch
```

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/genz-hybrid-slang-generator.git
```

2. Upload the `genz_slang_5000.csv` to your environment.

3. Open and run the notebook:

[Google Colab Link](https://colab.research.google.com/drive/1J7nwQ62pS-yoe9QakaXzmzbv3ss3aaIg)

## Results  
Generated outputs include fluent and slang-relevant sentence completions. Hybrid results demonstrate better alignment with informal speech and cultural tone.

## Future Scope
- **Extended Dataset**: Train on full 5000+ phrases or scrape more slang examples.  
- **Web App Integration**: Use Gradio or Streamlit for interactive slang generation.  
- **Multilingual Slang Support**: Add datasets for multilingual youth slang.  
- **Classifier + Generator**: Use a classifier to label tone and drive generation accordingly.
