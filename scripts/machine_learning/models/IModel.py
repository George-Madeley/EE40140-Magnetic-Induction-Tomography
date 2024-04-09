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