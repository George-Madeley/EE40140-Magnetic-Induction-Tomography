import pandas as pd

class MachineLearningModel:
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