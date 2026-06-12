import argparse
import time

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from data_utils import evaluate_predictions, load_chest_xray_data, print_metrics
from svm_from_scratch import SVMFromScratch


def train_sklearn_svm(X_train, y_train, max_iter: int):
    model = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, max_iter=max_iter, random_state=42),
    )
    model.fit(X_train, y_train)
    return model


def train_numpy_svm_for_comparison(X_train, y_train, n_iters: int):
    model = SVMFromScratch(learning_rate=0.0001, lambda_param=0.01, n_iters=n_iters)
    model.fit(X_train, y_train, verbose=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assignment 2: train and evaluate library SVM on Chest X-Ray data."
    )
    parser.add_argument("--max-iter", type=int, default=5000, help="LinearSVC max_iter value.")
    parser.add_argument(
        "--compare-numpy",
        action="store_true",
        help="Also train the NumPy SVM from assignment 1 and compare results.",
    )
    parser.add_argument(
        "--numpy-iters",
        type=int,
        default=1000,
        help="Number of epochs for the NumPy SVM comparison.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Optional limit per class for faster experiments.",
    )
    args = parser.parse_args()

    X_train, y_train = load_chest_xray_data("train", max_per_class=args.max_per_class)
    X_test, y_test = load_chest_xray_data("test", max_per_class=args.max_per_class)

    start_time = time.perf_counter()
    sklearn_model = train_sklearn_svm(X_train, y_train, max_iter=args.max_iter)
    sklearn_train_time = time.perf_counter() - start_time

    sklearn_predictions = sklearn_model.predict(X_test)
    sklearn_metrics = evaluate_predictions(y_test, sklearn_predictions)
    print_metrics("Assignment 2 - SVM implemented with sklearn", sklearn_metrics)
    print(f"- train_time_seconds: {sklearn_train_time:.2f}")

    if args.compare_numpy:
        start_time = time.perf_counter()
        numpy_model = train_numpy_svm_for_comparison(X_train, y_train, args.numpy_iters)
        numpy_train_time = time.perf_counter() - start_time

        numpy_predictions = numpy_model.predict(X_test)
        numpy_metrics = numpy_model.evaluate(y_test, numpy_predictions)
        print_metrics("Assignment 1 comparison - SVM implemented with NumPy", numpy_metrics)
        print(f"- train_time_seconds: {numpy_train_time:.2f}")

        print("\nMetric difference: sklearn - NumPy")
        for metric_name in sklearn_metrics:
            difference = sklearn_metrics[metric_name] - numpy_metrics[metric_name]
            print(f"- {metric_name}: {difference:+.4f}")


if __name__ == "__main__":
    main()
