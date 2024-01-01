import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf

root_dir = Path.cwd()
model_dir = root_dir.joinpath("artifacts")
data = pd.read_csv(root_dir.joinpath("laptopPrice.csv"))


class LogScaling(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        return np.log1p(X)


class TransformationPipeline:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        pass

    def preprocess(self):
        cat_cols = self.data.select_dtypes("object").columns
        num_cols = ["Number of Ratings", "Number of Reviews"]

        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            [
                ("log_transform", LogScaling(), num_cols),
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipelines", cat_pipeline, cat_cols),
            ],
            remainder="passthrough",
        )

        return preprocessor


class RegressorModel:
    def __init__(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.DataFrame,
        model_path: Path,
    ) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_path = model_path

    def model_train(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(79))
        model.add(tf.keras.layers.Dense(200))
        model.add(tf.keras.layers.Dense(200))
        model.add(tf.keras.layers.Dense(200))
        model.add(tf.keras.layers.Dense(1))
        model.compile(
            loss="mse",
            optimizer=tf.keras.optimizers.Adam(),
            metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
        )
        model.fit(
            self.X_train.toarray(),
            self.y_train,
            validation_data=(self.X_test.toarray(), self.y_test),
            epochs=20,
        )
        model.save(self.model_path)
        return


if __name__ == "__main__":
    X = data.drop(columns=["Price"])
    y = data["Price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    transformation = TransformationPipeline(X_train)
    preprocessor = transformation.preprocess()
    proc_obj = preprocessor.fit(X_train)
    joblib.dump(proc_obj, model_dir.joinpath("preprocessor.pkl"))
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    model_obj = RegressorModel(X_train, y_train, X_test, y_test, model_dir)
    model_obj.model_train()
