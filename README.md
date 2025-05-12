# BDA_project_Group10

## FunkSVD Variants for Movie Recommendation

This project explores different implementations of **FunkSVD** — a matrix factorization algorithm — applied to the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/). We compare versions trained using **Stochastic Gradient Descent (SGD)** and **Alternating Least Squares (ALS)**, and investigate the impact of including **bias terms** for users and items.

---

## Files in This Repo

| Notebook                              | Description                                                  |
|---------------------------------------|--------------------------------------------------------------|
| `funksvd_wbias.ipynb`                 | Implements FunkSVD using SGD, including bias terms.          |
| `funkSVD_sgd_no_bias.ipynb`           | Simplified FunkSVD using SGD without bias terms.             |
| `funkSVD_ALS.ipynb`                   | Implements matrix factorization using ALS instead of SGD.    |

---

## Models Implemented

- **FunkSVD with SGD (No Bias)**: Learns latent factors without user/item biases.
- **FunkSVD with SGD (With Bias)**: Adds user and item bias terms to the prediction formula.
- **FunkSVD with ALS**: Uses alternating optimization to learn factor matrices.

---

## Goals

- Implementation of funksvd algorithm
- Understand the impact of bias terms
- Compare convergence and RMSE across different training approaches
- Visualize overfitting trends using validation curves

---

## How to Run

Open each `.ipynb` file using Jupyter Notebook or VSCode.  
All datasets and code are self-contained.

---

## Group Members

- Yaqiao Huang
- Mert Kir 
- Sajjad Ayashm  
 

