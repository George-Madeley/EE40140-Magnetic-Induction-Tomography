from abc import ABC, abstractmethod

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