#!/usr/bin/env python
# coding: utf-8

import os
import os, sys
import argparse
import logging
import random

import coloredlogs
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc, confusion_matrix

import models
from dataset_utils import (
    Aggregate,
    extract_dataset,
    pad_and_fix,
    split_repos,
    split_into_x_and_y,
)
from helper import find_best_accuracy, find_best_f1, EnumAction, safe_mkdir

logger = logging.getLogger(__name__)
coloredlogs.install(fmt="%(asctime)s %(levelname)s %(message)s")


def model_selector(model_name, shape1, shape2, optimizer):
    return getattr(models, model_name)(shape1, shape2, optimizer)


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    exp_name,
    batch_size=32,
    epochs=20,
    model_name="conv1d",
    columns=None,
):
    if columns is None:
        columns = []

    optimizer = Adam(learning_rate=0.001)
    model = model_selector(model_name, X_train.shape[1], X_train.shape[2], optimizer)

    safe_mkdir("models")
    safe_mkdir(os.path.join("models", exp_name))
    safe_mkdir("figs")

    best_model_path = os.path.join("models", exp_name, f"{model_name}.keras")

    mcp_save = ModelCheckpoint(
        best_model_path,
        save_best_only=True,
        monitor="val_accuracy",
        mode="max",
    )

    es = EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        mode="max",
        restore_best_weights=True,
    )

    verbose = 1 if logger.level < logging.CRITICAL else 0
    history = model.fit(
        X_train,
        y_train,
        verbose=verbose,
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[mcp_save, es],
    )

    pyplot.figure()
    pyplot.plot(history.history["accuracy"])
    pyplot.plot(history.history["val_accuracy"])
    pyplot.title("model accuracy")
    pyplot.ylabel("accuracy")
    pyplot.xlabel("epoch")
    pyplot.legend(["train", "val"], loc="upper left")
    pyplot.savefig(os.path.join("figs", f"{exp_name}_{model_name}_accuracy.png"))
    pyplot.close()

    return tf.keras.models.load_model(
    best_model_path,
    custom_objects={"CenterTokenPooling": models.CenterTokenPooling},
)


def check_results(X_test, y_test, pred, model, exp_name, model_name, save=False):
    used_y_test = np.asarray(y_test).astype("float32")
    scores = model.evaluate(X_test, used_y_test, verbose=0)
    if len(scores) == 1:
        return 0

    max_f1, f1_thresh = find_best_f1(X_test, used_y_test, model)[:2]
    max_acc, acc_thresh, _ = find_best_accuracy(X_test, used_y_test, model)

    logger.critical(f"F1 - {max_f1}, {f1_thresh}")
    logger.critical(f"Acc - {max_acc}, {acc_thresh}")

    if save:
        safe_mkdir("results")
        with open(os.path.join("results", f"{exp_name}_{model_name}.txt"), "w") as mfile:
            mfile.write("Accuracy: %.2f%%\n" % (max_acc * 100))
            mfile.write("fscore: %.2f%%\n" % (max_f1 * 100))
            mfile.write("confusion matrix:\n")
            tn, fp, fn, tp = confusion_matrix(y_test, pred > acc_thresh).ravel()
            conf_matrix = f"tn={tn}, fp={fp}, fn={fn}, tp={tp}"
            mfile.write(conf_matrix)

        fpr = {}
        tpr = {}
        fpr["micro"], tpr["micro"], _ = roc_curve(used_y_test, pred)
        roc_auc = {"micro": auc(fpr["micro"], tpr["micro"])}

        pyplot.figure()
        lw = 2
        pyplot.plot(
            fpr["micro"],
            tpr["micro"],
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc["micro"],
        )
        pyplot.plot([0, 1], [0, 1], lw=lw, linestyle="--")
        pyplot.xlim([0.0, 1.0])
        pyplot.ylim([0.0, 1.05])
        pyplot.xlabel("False Positive Rate")
        pyplot.ylabel("True Positive Rate")
        pyplot.title("Receiver operating characteristic")
        pyplot.legend(loc="lower right")
        pyplot.savefig(os.path.join("figs", f"auc_{exp_name}_{model_name}.png"))
        pyplot.close()

    return max_acc


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--hours", type=int, default=0, help="hours back")
    parser.add_argument("-d", "--days", type=int, default=10, help="days back")
    parser.add_argument("--resample", type=int, default=24, help="hours for time aggregation")
    parser.add_argument("-r", "--ratio", type=int, default=1, help="benign:vuln ratio")
    parser.add_argument("-a", "--aggr", type=Aggregate, action=EnumAction, default=Aggregate.before_cve)
    parser.add_argument("-b", "--backs", type=int, default=10, help="event window size parameter")
    parser.add_argument("-v", "--verbose", action="store_const", dest="loglevel", const=logging.DEBUG)
    parser.add_argument("-c", "--cache", "--cached", action="store_const", dest="cache", const=True)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-m", "--model", action="store", type=str, default="conv1d")
    parser.add_argument("-k", "--kfold", type=int, default=10)
    parser.add_argument("--comment", action="store", type=str, default="")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--metadata", action="store_true")
    parser.add_argument("--data-location", action="store", default="data_collection")
    parser.add_argument("--cache-location", action="store", default="ready_data")
    return parser.parse_args()


def init():
    seed = 555

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # force deterministic ops where possible
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def main():
    args = parse_args()
    logger.level = args.loglevel or logging.CRITICAL
    init()


    print("RUNNING FILE =", os.path.abspath(__file__))
    print("ARGV =", sys.argv)
    print("DATA LOCATION =", args.data_location)

    all_repos, exp_name, columns = extract_dataset(
        aggr_options=args.aggr,
        resample=args.resample,
        benign_vuln_ratio=args.ratio,
        hours=args.hours,
        days=args.days,
        backs=args.backs,
        cache=args.cache,
        metadata=args.metadata,
        comment=args.comment,
        data_location=args.data_location,
        cache_location=args.cache_location,
    )

    all_repos, num_of_vulns = pad_and_fix(all_repos)

    train_size = int(0.8 * num_of_vulns)
    validation_size = int(0.1 * num_of_vulns)
    _test_size = num_of_vulns - train_size - validation_size

    logger.info(f"Train size: {train_size}")
    logger.info(f"Validation size: {validation_size}")
    logger.info(f"Test size: {_test_size}")

    train_and_val_repos, test_repos, _ = split_repos(all_repos, train_size + validation_size)

    best_model = None
    best_val_accuracy = 0

    for _ in range(args.kfold):
        train_repos, val_repos, num_of_train_repos = split_repos(train_and_val_repos, train_size)

        if not train_repos or not val_repos:
            raise RuntimeError("Train/validation repository split produced an empty partition.")

        X_train, y_train = split_into_x_and_y(train_repos)
        X_val, y_val = split_into_x_and_y(val_repos)

        model = train_model(
            X_train,
            y_train,
            X_val,
            y_val,
            exp_name,
            batch_size=args.batch,
            epochs=args.epochs,
            model_name=args.model,
            columns=columns,
        )

        pred = model.predict(X_val, verbose=0).reshape(-1)
        acc = check_results(X_val, y_val, pred, model, exp_name, args.model)

        if acc > best_val_accuracy:
            best_model = model
            best_val_accuracy = acc

        num_of_val_repos = len(train_and_val_repos) - num_of_train_repos
        train_and_val_repos = train_and_val_repos[-num_of_val_repos:] + train_and_val_repos[:-num_of_val_repos]

    if best_model is None:
        raise RuntimeError("No model was successfully trained.")

    X_test, y_test = split_into_x_and_y(test_repos)
    pred = best_model.predict(X_test, verbose=0).reshape(-1)
    acc = check_results(X_test, y_test, pred, best_model, exp_name, args.model, save=True)
    logging.critical(f"Best val accuracy: {best_val_accuracy}")
    logging.critical(f"Best test accuracy: {acc}")


if __name__ == "__main__":
    main()