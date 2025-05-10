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

## Goals

- Implementation of funksvd algorithm
- Understand the impact of bias terms
- Compare convergence and RMSE across different training approaches
- Visualize overfitting trends using validation curves

## Group Members

- Sajjad Ayashm  
- Mert Kr  
- Yaqiao Huang
