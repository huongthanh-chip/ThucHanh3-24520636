from data_utils import load_chest_xray_data, print_metrics
from svm_from_scratch import SVMFromScratch


def main() -> None:
    X_train, y_train = load_chest_xray_data("train")
    X_test, y_test = load_chest_xray_data("test")

    model = SVMFromScratch(learning_rate=0.0001, lambda_param=0.01, n_iters=1000)
    model.fit(X_train, y_train, verbose=True)

    y_predict = model.predict(X_test)
    metrics = model.evaluate(y_test, y_predict)
    print_metrics("Assignment 1 - SVM implemented with NumPy", metrics)


if __name__ == "__main__":
    main()
