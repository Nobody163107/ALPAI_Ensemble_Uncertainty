"""Run all four experiments in order and regenerate every figure and metrics file.

    python experiments/run_all.py            # E4 reuses the trained CNNs in checkpoints/
    python experiments/run_all.py --retrain  # also retrain the 11 CNNs (about 25 min on a CPU)
"""
import argparse
import time

import e1_toy_classification
import e2_regression
import e3_accuracy_rejection
import e4_mnist_ood

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="retrain the MNIST CNNs instead of loading them")
    args = parser.parse_args()
    t0 = time.time()
    for name, step in [("E1 2D toy classification", e1_toy_classification.run),
                       ("E2 regression", e2_regression.run),
                       ("E3 accuracy-rejection curves", e3_accuracy_rejection.run),
                       ("E4 MNIST / Fashion-MNIST", lambda: e4_mnist_ood.run(retrain=args.retrain))]:
        print(f"\n===== {name} =====")
        step()
    print(f"\nAll experiments done in {(time.time() - t0) / 60:.1f} min. Figures: results/figures, numbers: results/metrics")
