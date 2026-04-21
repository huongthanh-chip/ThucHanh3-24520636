import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

class SVM:
    def __init__(self, learning_rate=0.0001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param 
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.history_loss = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # y_hat ở đây chính là: np.dot(x_i, self.w) + self.b
                # Đây là giá trị dự đoán thô (raw score)
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                
                if condition:
                    # Đạo hàm phần Regularization: d/dw [lambda * ||w||^2] = 2 * lambda * w
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Đạo hàm phần Hinge Loss + Regularization
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.outer(y[idx], x_i).flatten())
                    self.b -= self.lr * (-y[idx])
            
            # Tính loss tổng sau mỗi epoch để theo dõi
            current_loss = self.loss_fn(X, y)
            self.history_loss.append(current_loss)

    def predict(self, X):
        approx = np.dot(X, self.w) + self.b
        return np.sign(approx)
    
    def loss_fn(self, X, y):
        # y_hat = w.X + b (giá trị dự đoán thô)
        y_hat = np.dot(X, self.w) + self.b
        # Công thức Hinge Loss: 1/2 * ||w||^2 + C * sum(max(0, 1 - y * y_hat))
        # Ở đây lambda_param đóng vai trò điều chỉnh tỷ lệ
        reg_term = self.lambda_param * np.sum(self.w**2)
        hinge_term = np.mean(np.maximum(0, 1 - y * y_hat))
        return reg_term + hinge_term
    