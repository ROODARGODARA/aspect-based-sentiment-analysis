# Multimodal Sentiment Analysis

This project implements a **multimodal sentiment analysis** system that combines
textual and visual information to classify sentiment as **positive, neutral, or negative**.
The primary goal is to demonstrate an end-to-end multimodal pipeline using modern
deep learning models.

The implementation is designed for **Google Colab** and is intended for academic
and learning purposes.

---

## Project Overview

Traditional sentiment analysis relies only on text, which can miss important
context available in images. This project integrates both modalities:

- **Text** is encoded using **BERT (bert-base-uncased)**
- **Images** are encoded using **ResNet-50 (ImageNet pretrained)**
- Features from both modalities are **fused at feature level**
- A classifier predicts sentiment labels: Positive, Neutral, or Negative

The project focuses on **correctness of the pipeline and architecture**, not on
benchmarking real-world performance.

---

## Repository Structure

multimodal-sentiment-analysis/
├── notebooks/
│ └── multimodal_sentiment_analysis.ipynb
├── train_clean.csv
├── images/
│ └── .gitkeep
├── requirements.txt
└── README.md


---

## Dataset Format

The model expects a CSV file with the following columns:

| Column | Description |
|------|------------|
| `text` | Text content (e.g., tweet or caption) |
| `image` | Image filename |
| `label` | Sentiment label (0, 1, 2) |

Label encoding:
- `0` → Negative  
- `1` → Neutral  
- `2` → Positive  

> Note: In this project, individual `.txt` files were consolidated into a single
> CSV (`train_clean.csv`) as part of preprocessing.

---

## Implementation Details

- **Text Encoder:** BERT (768-dimensional embeddings)
- **Image Encoder:** ResNet-50 (2048-dimensional embeddings)
- **Fusion:** Concatenation after linear projection
- **Classifier:** Fully connected layer with softmax output
- **Loss Function:** Cross-Entropy Loss
- **Optimizer:** Adam (learning rate = 1e-4)

---

## How to Run (Google Colab)

1. Open the notebook:
notebooks/multimodal_sentiment_analysis.ipynb

2. Upload the following to Colab:
- `train_clean.csv`
- `images/` directory (or sample images)
3. Install dependencies (handled in the notebook)
4. Run all cells sequentially

---

## Results and Observations

Due to the **small demonstration dataset** and evaluation on training data,
the model achieves high accuracy. This is expected and is used to validate
the correctness of the multimodal training pipeline rather than generalization
performance.

---

## Notes for Evaluation

- The notebook is intended to be executed in **Google Colab**
- GitHub preview issues may occur due to Colab widget metadata
- The **Code tab** and **Open in Colab** option should be used for review

---

## Future Work

- Aspect term extraction
- Cross-modal attention mechanisms
- Larger annotated multimodal datasets
- Proper train/validation/test splits

---

## License

This project is intended for academic use.
