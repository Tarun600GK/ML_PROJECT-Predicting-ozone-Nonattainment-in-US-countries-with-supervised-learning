import argparse
import os

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

FEATURES = ["temperature", "pressure", "wind", "month", "lat", "lon", "no2", "voc"]
TARGET = "ozone"


def split(df):
    train = df[df["date"] < "2017-01-01"]
    val = df[(df["date"] >= "2017-01-01") & (df["date"] < "2017-07-01")]
    test = df[df["date"] >= "2017-07-01"]
    return train, val, test


def get_models():
    return {
        "linear_regression": LinearRegression(),
        "svr_paper": SVR(kernel="poly", epsilon=0.1),
        "svr_tuned": SVR(kernel="rbf", epsilon=0.002, C=1),
        "neural_network": MLPRegressor(hidden_layer_sizes=(200, 200, 200), learning_rate="adaptive",
                                       early_stopping=True, max_iter=200, random_state=0),
        "gradient_boosting": HistGradientBoostingRegressor(max_iter=500, learning_rate=0.1, random_state=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ozone.csv")
    parser.add_argument("--out", default="results")
    parser.add_argument("--svr-samples", type=int, default=20000)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data, parse_dates=["date"])
    train, val, test = split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    scaler = MinMaxScaler().fit(train[FEATURES])
    X = {name: scaler.transform(part[FEATURES]) for name, part in
         [("train", train), ("val", val), ("test", test)]}
    y = {"train": train[TARGET].values, "val": val[TARGET].values, "test": test[TARGET].values}

    rows = []
    for name, model in get_models().items():
        Xtr, ytr = X["train"], y["train"]
        if name.startswith("svr") and len(ytr) > args.svr_samples:
            idx = pd.Series(range(len(ytr))).sample(args.svr_samples, random_state=0).values
            Xtr, ytr = Xtr[idx], ytr[idx]
        print("training", name)
        model.fit(Xtr, ytr)
        row = {"model": name}
        for part in ["train", "val", "test"]:
            pred = model.predict(X[part])
            row[f"r2_{part}"] = round(r2_score(y[part], pred), 4)
            row[f"mae_{part}"] = round(mean_absolute_error(y[part], pred), 5)
        print(row)
        rows.append(row)
        joblib.dump(model, os.path.join(args.out, f"{name}.joblib"))

    joblib.dump(scaler, os.path.join(args.out, "scaler.joblib"))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
