from glob import glob
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, TensorDataset


def calculate_f1_score(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    TP = matrix[0][0]
    FP = matrix[1][0]
    FN = matrix[0][1]
    P = TP + FN
    recall = TP / P
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def custom_calculate_f1_score(estimator, X, y):
    y_pred = estimator.predict(X)
    return calculate_f1_score(y, y_pred)


def get_all_metrics(y_test, y_pred):
    matrix = confusion_matrix(y_test, y_pred)
    TP = matrix[0][0]
    TN = matrix[1][1]
    FP = matrix[1][0]
    FN = matrix[0][1]
    P = TP + FN
    N = TN + FP
    TPR = TP / P
    TNR = TN / N
    FPR = FP / N
    FNR = FN / P
    recall = TPR
    precision = TP / (TP + FP)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (TP + TN) / (P + N)
    error_rate = (FP + FN) / (P + N)
    balanced_accuracy = (TPR + TNR) / 2
    true_skill_statistics = TPR - FPR
    heidke_skill_score = (TP) / (TP + FN) - (FP) / (FP + TN)

    return {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "P": P,
        "N": N,
        "TPR": TPR,
        "TNR": TNR,
        "FPR": FPR,
        "FNR": FNR,
        "Recall": recall,
        "Precision": precision,
        "F1 Score": f1_score,
        "Accuracy": accuracy,
        "Error Rate": error_rate,
        "Balanced Accuracy": balanced_accuracy,
        "True Skill Statistics": true_skill_statistics,
        "Heidke Skill Score": heidke_skill_score,
    }


class TorchKFoldCrossValidation:
    def __init__(
        self,
        model_class,
        loss_fn,
        learning_rate,
        epochs,
        batch_size,
        cv: StratifiedShuffleSplit,
        device,
    ) -> None:
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.cv = cv
        self.device = device
        self.models = []

    def training_loop(self, train_loader, model, optimizer):
        model.train()
        for X_batch, y_batch in train_loader:
            pred = model(X_batch)
            loss = self.loss_fn(pred, y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            pred = model(train_loader.dataset.tensors[0])
            y_train = train_loader.dataset.tensors[1]

            f1_score = calculate_f1_score(
                torch.argmax(y_train, dim=1).cpu().numpy(),
                torch.argmax(pred, dim=1).cpu().numpy(),
            )
            # print(f"Training metrics")
            # print(f"Loss: {loss.item()}")
            # print(f"F1 Score: {f1_score}")

    def val_nn_model(self, val_dataset: TensorDataset, model):
        model.eval()
        with torch.no_grad():
            X_val, y_val = val_dataset.tensors
            pred = model(X_val)
            loss = self.loss_fn(pred, y_val)

            f1_score = calculate_f1_score(
                torch.argmax(y_val, dim=1).cpu().numpy(),
                torch.argmax(pred, dim=1).cpu().numpy(),
            )
            # print(f"Validation metrics")
            # print(f"Loss: {loss.item()}")
            # print(f"F1 Score: {f1_score}")
            return loss, pred, f1_score

    def fit(self, X, y):
        val_cv = {}
        for cv_step, (train_index, val_index) in enumerate(self.cv.split(X, y)):
            best_f1_score = -1
            print(f"Cross validation step {cv_step+1}\n")

            model = self.model_class()  # .to(self.device)
            optimizer = torch.optim.Adamax(model.parameters(), lr=self.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.45
            )

            X_train = torch.tensor(
                X[train_index], dtype=torch.float32
            )  # .to(self.device)
            y_train = torch.tensor(
                y[train_index], dtype=torch.float32
            )  # .to(self.device)

            X_val = torch.tensor(X[val_index], dtype=torch.float32)  # .to(self.device)
            y_val = torch.tensor(y[val_index], dtype=torch.float32)  # .to(self.device)

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_dataset = TensorDataset(X_val, y_val)

            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                self.training_loop(train_loader, model, optimizer)
                lr_scheduler.step()

                _, _, val_f1_score = self.val_nn_model(val_dataset, model)
                if val_f1_score > best_f1_score:
                    best_f1_score = val_f1_score
                    torch.save(model, f"models/model_cv_{cv_step+1}.pth")
                    print(f"Model saved with F1 Score: {best_f1_score}")

                print(f"Epoch {epoch + 1} completed\n")

            _, y_pred, _ = self.val_nn_model(val_dataset, model)
            val_cv[cv_step + 1] = get_all_metrics(
                y_val.argmax(dim=1), y_pred.argmax(dim=1)
            )
            val_cv[cv_step + 1]["Brier Score"] = (
                torch.mean((y_pred[:, 1] - y_val.argmax(dim=1)) ** 2).cpu().numpy()
            )
            val_cv[cv_step + 1]["Brier Skill Score"] = (
                (
                    val_cv[cv_step + 1]["Brier Score"]
                    / (
                        torch.mean(
                            (y_val.argmax(dim=1) - torch.mean(y_pred[:, 1])) ** 2
                        )
                    )
                )
                .cpu()
                .numpy()
            )
            self.models.append(model)

        val_cv["mean"] = pd.DataFrame(val_cv).mean(axis=1)
        return pd.DataFrame(val_cv).round(4)

    def load_models(self):
        for path in glob(os.path.join("models", "model_cv_*.pth")):
            print(f"Loading model from {path}")
            self.models.append(torch.load(path))

    def predict(self, X):
        preds = []
        for cv_model in self.models:
            cv_model.eval()
            with torch.no_grad():
                preds.append(cv_model(X).cpu())
        preds = np.array(preds)
        return np.mean(preds, axis=0)
