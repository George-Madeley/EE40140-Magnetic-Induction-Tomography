from abc import ABC, abstractmethod
from sys import _getframe
from typing import Literal, get_args, get_origin

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class IModel(ABC):
    @abstractmethod
    def train(self, train_df):
        """
        Train the model
        
        :param train_df: training dataframe
        """
        pass

    @abstractmethod
    def test(self, test_df):
        """
        Test the model
        
        :param test_df: test dataframe
        
        :return: metrics
        """
        pass

    @abstractmethod
    def predict(self, predict_df):
        """
        Predict the labels of the test data
        
        :param predict_df: prediction dataframe
        
        :return: predictions
        """
        pass

    @abstractmethod
    def isParamValid(self, name, value):
        """
        Check if the parameter is valid
        
        :param name: name of the parameter
        :param value: value of the parameter
        
        :return: boolean
        """
        pass

    @abstractmethod
    def getDefaultParams(self):
        """
        Get the default parameters
        
        :return: default parameters
        """
        pass

    
    @staticmethod
    def enforceLiterals(function):
        kwargs = _getframe(1).f_locals
        for name, type_ in function.__annotations__.items():
            value = kwargs.get(name)
            options = get_args(type_)
            if get_origin(type_) is Literal and name in kwargs and value not in options:
                raise AssertionError(f"'{value}' is not in {options} for '{name}'")

    def getLabels(self, df):
        """
        Get the labels from the dataframe
        
        :param df: dataframe
        
        :return: labels
        """
        return df['shape'].values
    
    def getValuesWithBackgroundNoise(self, df):
        """
        Get the values with background noise
        
        :param df: dataframe
        
        :return: values
        """
        # Get the values of the background noise
        bbColumnNames = df.filter(regex='^bb_\d{1,3}$').columns
        bbValues = df[bbColumnNames].values

        # Get the values of the sample
        ccColumnNames = df.filter(regex='^cc_\d{1,3}$').columns
        ccValues = df[ccColumnNames].values

        # Combine the values of the background noise and the sample to the new
        # dataframe has a total of 240 columns
        values = pd.concat([bbValues, ccValues], axis=1)

        return values
    
    def getValuesWithoutBackgroundNoise(self, df):
        """
        Get the values without background noise
        
        :param df: dataframe

        :return: values
        """

        # Get the values of the sample
        ccColumnNames = df.filter(regex='^cc_\d{1,3}$').columns
        ccValues = df[ccColumnNames].values

        return ccValues
    
    def evaluate(self, true_labels, predicted_labels) -> dict:
        metrics = {
            'accuracy': accuracy_score(true_labels, predicted_labels),
            'precision': precision_score(true_labels, predicted_labels),
            'recall': recall_score(true_labels, predicted_labels),
            'f1': f1_score(true_labels, predicted_labels),
            'roc_auc': roc_auc_score(true_labels, predicted_labels)
        }

        return metrics