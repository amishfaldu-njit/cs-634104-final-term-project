import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import DataLoader, TensorDataset


def f1_score_weighted(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    precision = np.diag(matrix) / matrix.sum(axis=0)
    recall = np.diag(matrix) / matrix.sum(axis=1)
    f1 = 2 * precision * recall / (precision + recall)
    weighted_f1 = np.sum(f1 * matrix.sum(axis=1) / matrix.sum())
    return weighted_f1


def custom_f1_scoring_fn(estimator, X, y):
    y_pred = estimator.predict(X)
    return f1_score_weighted(y, y_pred)


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

            f1_score = f1_score_weighted(
                torch.argmax(y_train, dim=1).cpu().numpy(),
                torch.argmax(pred, dim=1).cpu().numpy(),
            )
            print(f"Training metrics")
            print(f"Loss: {loss.item()}")
            print(f"F1 Score: {f1_score}")

    def val_nn_model(self, val_dataset: TensorDataset, model):
        model.eval()
        with torch.no_grad():
            X_val, y_val = val_dataset.tensors
            pred = model(X_val)
            loss = self.loss_fn(pred, y_val)

            f1_score = f1_score_weighted(
                torch.argmax(y_val, dim=1).cpu().numpy(),
                torch.argmax(pred, dim=1).cpu().numpy(),
            )
            print(f"Validation metrics")
            print(f"Loss: {loss.item()}")
            print(f"F1 Score: {f1_score}")
            return f1_score

    def fit(self, X, y):
        for cv_step, (train_index, val_index) in enumerate(self.cv.split(X, y)):
            print(f"Cross validation step {cv_step+1}\n")

            model = self.model_class() #.to(self.device)
            optimizer = torch.optim.Adamax(model.parameters(), lr=self.learning_rate)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.45
            )

            X_train = torch.tensor(X[train_index], dtype=torch.float32) #.to(self.device)
            y_train = torch.tensor(y[train_index], dtype=torch.float32) #.to(self.device)

            X_val = torch.tensor(X[val_index], dtype=torch.float32) #.to(self.device)
            y_val = torch.tensor(y[val_index], dtype=torch.float32) #.to(self.device)

            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_dataset = TensorDataset(X_val, y_val)

            for epoch in range(self.epochs):
                print(f"Epoch {epoch + 1}\n-------------------------------")
                self.training_loop(train_loader, model, optimizer)
                lr_scheduler.step()
                print()

                self.val_nn_model(val_dataset, model)
                print()

            self.models.append(model)

            print()

    def predict(self, X):
        preds = []
        for cv_model in self.models:
            cv_model.eval()
            with torch.no_grad():
                preds.append(cv_model(X).cpu())
        preds = np.array(preds)
        print(preds.shape)
        return np.mean(preds, axis=0)
