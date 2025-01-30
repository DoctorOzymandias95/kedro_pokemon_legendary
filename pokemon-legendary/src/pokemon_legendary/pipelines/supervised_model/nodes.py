"""
This is a boilerplate pipeline 'supervised_model'
generated using Kedro 0.19.11
"""

from typing import Tuple, Any, Dict
import logging 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sn

logger = logging.getLogger(__name__)

def data_preprocessing(poke_df: pd.DataFrame) -> Tuple:
     
	### Drop unnecessary columns
	poke_df = poke_df.drop(columns=["#", "Name"])
	### Renaming columns to remove spaces
	poke_df = poke_df.rename(columns = {element: element.replace(" ", "_") for element in poke_df.columns})

	### Split X dataset and target variable
	X_columns = list(filter(lambda _: 'Legendary' not in _, poke_df.columns))
	X_dataset = poke_df[X_columns]
	y_dataset = poke_df['Legendary']

	### Filling missing categorical values
	cat_impute_type_2 = SimpleImputer(strategy='constant', fill_value = 'Base')
	X_dataset["Type_2"] = cat_impute_type_2.fit_transform([X_dataset["Type_2"].to_list()])[0]
	
	### Encoding categorical features
	categorical_cols = [col for col in X_dataset.columns if X_dataset[col].dtypes not in ["int64", "float64"]]
	X_dataset = pd.get_dummies(X_dataset, columns=categorical_cols)

	### Standardizing numerical features
	std_scaler = StandardScaler()
	numerical_cols = [col for col in X_dataset.columns if X_dataset[col].dtypes in ["int64", "float64"]]
	X_dataset[numerical_cols] = std_scaler.fit_transform(X_dataset[numerical_cols])
     
	logger.info(">> Preprocessed Input data")

	return X_dataset, y_dataset

"""----------------------------------------"""

def split_data(X_data: pd.DataFrame, y_data: pd.Series, parameters: dict[str, Any]) -> Tuple:
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=parameters["split_test_size"], random_state=parameters["random_state"]
    )
    logger.info(">> Input data split for model train")
    return X_train, X_test, y_train, y_test

"""----------------------------------------"""

def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict[str, Any]) -> GradientBoostingClassifier:
    
    boost_model = GradientBoostingClassifier(n_estimators=parameters["boost_n_estimators"], random_state=parameters["random_state"])
    boost_model.fit(X_train, y_train)
    logger.info(">> Gradient Boosting Classifier trained")
    return boost_model

"""----------------------------------------"""

def evaluate_model(boosting: GradientBoostingClassifier, X_test: pd.DataFrame, y_test: pd.Series)-> Tuple[pd.DataFrame, Dict[str, float]]:

    y_pred = boosting.predict(X_test)
    classification_rep = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    f1_ = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    logger.info(">> Model evaluated")
    return classification_rep, {"f1_score": f1_, "precision": precision, "recall": recall}

"""----------------------------------------"""

def create_conf_matrix(boosting: GradientBoostingClassifier, X_test: pd.DataFrame, y_test: pd.Series):
     
     y_pred = boosting.predict(X_test)
     data = {"y_actual": y_test.Legendary.values, "y_predicted": y_pred}
     df = pd.DataFrame(data, columns=["y_actual", "y_predicted"])
     conf_mtrx = pd.crosstab(
        df["y_actual"], df["y_predicted"], rownames=["Actual"], colnames=["Predicted"]
    )
     sn.heatmap(conf_mtrx, annot=True)
     return plt
    

