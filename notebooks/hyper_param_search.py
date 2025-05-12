import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split

def grid_search_funksvd(df,
                        lr_list,
                        lambda_list,
                        K_list,
                        n_epochs=20,
                        test_size=0.2,
                        random_state=42,
                        output_csv='hyperparam_results.csv'):
    """
    Perform grid search over (learning rate, regularization, factor number)
    for FunkSVD (no bias) trained by SGD, and save results to CSV.

    Args:
        df (pd.DataFrame): must contain ['userId','movieId','rating'].
        lr_list (list of float): candidate learning rates.
        lambda_list (list of float): candidate regularization strengths.
        K_list (list of int): candidate latent factor dimensions.
        n_epochs (int): number of SGD passes per run.
        test_size (float): fraction of data for test split.
        random_state (int): for reproducibility of train/test split.
        output_csv (str): path to save results CSV.

    Returns:
        pd.DataFrame: columns = ['lr','lambda','K','train_rmse','test_rmse'].
    """
    # 1) train/test split
    df_small = df[['userId','movieId','rating']]
    trainval_df, test_df = train_test_split(df_small,
                                         test_size=test_size,
                                         random_state=random_state)
    train_df, val_df = train_test_split(trainval_df, test_size=0.2, random_state=42)

    # 2) build index mappings once
    user_ids = df_small['userId'].unique()
    movie_ids = df_small['movieId'].unique()
    u2idx = {u:i for i,u in enumerate(user_ids)}
    i2idx = {m:i for i,m in enumerate(movie_ids)}
    n_users, n_items = len(user_ids), len(movie_ids)

    # 3) build sample lists
    train_samples = [(u2idx[u], i2idx[m], r)
                     for u, m, r in zip(train_df.userId,
                                        train_df.movieId,
                                        train_df.rating)]
    val_samples  = []
    for u, m, r in zip(val_df.userId,
                       val_df.movieId,
                       val_df.rating):
        if u in u2idx and m in i2idx:
            val_samples.append((u2idx[u], i2idx[m], r))

    results = []
    # 4) grid search
    for lr, lmbda, K in itertools.product(lr_list, lambda_list, K_list):
        # initialize P, Q
        P = np.random.normal(scale=0.01, size=(n_users, K))
        Q = np.random.normal(scale=0.01, size=(n_items, K))
        # SGD
        for epoch in range(n_epochs):
            np.random.shuffle(train_samples)
            for u_idx, i_idx, r in train_samples:
                pred = P[u_idx].dot(Q[i_idx])
                err  = r - pred
                P[u_idx] += lr * (err * Q[i_idx] - lmbda * P[u_idx])
                Q[i_idx] += lr * (err * P[u_idx] - lmbda * Q[i_idx])

        # evaluate train RMSE
        train_se = sum((r - P[u].dot(Q[i]))**2 for u, i, r in train_samples)
        train_rmse = np.sqrt(train_se / len(train_samples))
        # evaluate test RMSE
        val_se  = sum((r - P[u].dot(Q[i]))**2 for u, i, r in val_samples)
        val_rmse = np.sqrt(val_se / len(val_samples))

        results.append({
            'lr': lr,
            'lambda': lmbda,
            'K': K,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse
        })

    # 5) save to CSV and return DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"✔️ Grid search done, results saved to '{output_csv}'")
    return results_df

