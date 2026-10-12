import argparse
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from train import FEATURES, TARGET, split

BLUE = "#2a78d6"
GREY = "#52514e"
NAMES = {
    "linear_regression": "Linear Regression",
    "svr_paper": "SVR (paper, eps=0.1)",
    "svr_tuned": "SVR (RBF, eps=0.002)",
    "neural_network": "Neural Network",
    "gradient_boosting": "Gradient Boosting",
}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GREY)
    ax.tick_params(colors=GREY)
    ax.grid(alpha=0.2)


def plot_predictions(models, X, y, out):
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4), sharey=True)
    lim = (0, max(y.max(), 0.08))
    for ax, (name, model) in zip(axes, models.items()):
        ax.scatter(y, model.predict(X), s=8, alpha=0.4, color=BLUE, edgecolors="none")
        ax.plot(lim, lim, color=GREY, lw=1, ls="--")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(NAMES[name], fontsize=11)
        ax.set_xlabel("True ozone (ppm)")
        style(ax)
    axes[0].set_ylabel("Predicted ozone (ppm)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "pred_vs_true.png"), dpi=150)
    plt.close(fig)


def plot_r2(metrics, out):
    m = metrics.set_index("model")
    labels = [NAMES[n] for n in m.index]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    vals = m["r2_test"].clip(lower=0)
    ax.barh(labels, vals, color=BLUE, height=0.6)
    for i, v in enumerate(m["r2_test"]):
        ax.text(max(v, 0) + 0.01, i, f"{v:.3f}", va="center", color=GREY, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Test R² (negative values shown as 0)")
    ax.invert_yaxis()
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "r2_test.png"), dpi=150)
    plt.close(fig)


def plot_importance(model, X, y, out):
    result = permutation_importance(model, X, y, n_repeats=5, random_state=0, scoring="r2")
    imp = pd.Series(result.importances_mean, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
    ax.set_xlabel("Drop in test R² when feature is shuffled")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "feature_importance.png"), dpi=150)
    plt.close(fig)
    return imp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ozone.csv")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()

    df = pd.read_csv(args.data, parse_dates=["date"])
    _, val, test = split(df)
    scaler = joblib.load(os.path.join(args.results, "scaler.joblib"))
    metrics = pd.read_csv(os.path.join(args.results, "metrics.csv"))
    models = {n: joblib.load(os.path.join(args.results, f"{n}.joblib")) for n in metrics["model"]}

    plot_predictions(models, scaler.transform(val[FEATURES]), val[TARGET].values, args.results)
    plot_r2(metrics, args.results)

    best = metrics.sort_values("r2_val", ascending=False)["model"].iloc[0]
    imp = plot_importance(models[best], scaler.transform(test[FEATURES]), test[TARGET].values, args.results)
    print("best model:", best)
    print(imp.sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
