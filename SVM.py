import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

class SVM:
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param 
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.history_loss = []

    def fit(self, X, y, verbose=True, log_every=50):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.history_loss = []
        epoch_pbar = tqdm(range(1, self.n_iters + 1), desc="Training SVM", disable=not verbose)

        for epoch in epoch_pbar:
            for idx, x_i in enumerate(X):
                # y_hat ở đây chính là: np.dot(x_i, self.w) + self.b
                # Đây là giá trị dự đoán thô (raw score)
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # Đạo hàm phần Regularization: d/dw [lambda * ||w||^2] = 2 * lambda * w
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Đạo hàm phần Hinge Loss + Regularization
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y[idx] * x_i)
                    self.b += self.lr * y[idx]
            
            # Tính loss tổng sau mỗi epoch để theo dõi
            current_loss = self.loss_fn(X, y)
            self.history_loss.append(current_loss)

            if verbose and epoch % log_every == 0:
                train_pred = self.predict(X)
                train_acc = np.mean(train_pred == y)
                # set_postfix sẽ đẩy thông tin ra phía sau thanh progress
                epoch_pbar.set_postfix({
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{train_acc:.4f}"
                })

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        y_hat = np.sign(approx)
        y_hat[y_hat == 0] = 1
        return y_hat
    
    def loss_fn(self, X, y):
        # y_hat = w.X + b (giá trị dự đoán thô)
        y_hat = np.dot(X, self.w) + self.b
        # Công thức Hinge Loss: 1/2 * ||w||^2 + C * sum(max(0, 1 - y * y_hat))
        # Ở đây lambda_param đóng vai trò điều chỉnh tỷ lệ
        reg_term = self.lambda_param * np.sum(self.w**2)
        hinge_term = np.mean(np.maximum(0, 1 - y * y_hat))
        return reg_term + hinge_term
    
    def evaluate(self, y, y_hat) -> dict:
        accuracy = np.mean(y == y_hat)
        precision = precision_score(y, y_hat, average='weighted', zero_division=0)
        recall = recall_score(y, y_hat, average='weighted', zero_division=0)
        f1 = f1_score(y, y_hat, average='weighted', zero_division=0)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }