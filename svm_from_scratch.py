import numpy as np
from tqdm import tqdm

from data_utils import evaluate_predictions


class SVMFromScratch:
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.history_loss = []

    def fit(self, X, y, verbose=True, log_every=50):
        _, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history_loss = []

        epoch_pbar = tqdm(
            range(1, self.n_iters + 1),
            desc="Training SVM from scratch",
            disable=not verbose,
        )

        for epoch in epoch_pbar:
            for idx, x_i in enumerate(X):
                raw_score = np.dot(x_i, self.w) + self.b
                condition = y[idx] * raw_score >= 1

                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y[idx] * x_i)
                    self.b += self.lr * y[idx]

            current_loss = self.loss_fn(X, y)
            self.history_loss.append(current_loss)

            if verbose and epoch % log_every == 0:
                train_pred = self.predict(X)
                train_acc = np.mean(train_pred == y)
                epoch_pbar.set_postfix(
                    {
                        "loss": f"{current_loss:.4f}",
                        "acc": f"{train_acc:.4f}",
                    }
                )

    def predict(self, X):
        raw_score = np.dot(X, self.w) + self.b
        y_hat = np.sign(raw_score)
        y_hat[y_hat == 0] = 1
        return y_hat.astype(np.int32)

    def loss_fn(self, X, y):
        raw_score = np.dot(X, self.w) + self.b
        regularization_loss = self.lambda_param * np.sum(self.w**2)
        hinge_loss = np.mean(np.maximum(0, 1 - y * raw_score))
        return regularization_loss + hinge_loss

    def evaluate(self, y, y_hat) -> dict[str, float]:
        return evaluate_predictions(y, y_hat)
