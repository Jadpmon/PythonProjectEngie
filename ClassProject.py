import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

class ExploreData:
    dataframe = None
    dataframe_path = None

    def __init__(self, dataframe=None, dataframe_path=None):
        if dataframe is not None:
            self.dataframe = dataframe
        if dataframe_path is not None:
            self.dataframe_path: dataframe_path
            self.dataframe = pd.read_csv(self.dataframe_path)

    def info_data(self):
        print( ' ** format du dataframe (ligne,colonne) = {} ** \n'.format(self.dataframe.shape))
        print('** description du data frame **')
        print(self.dataframe.describe())

        def Missing_values(data):
            total = data.isnull().sum().sort_values(ascending=False)
            percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
            missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Pourcentage'])
            # Affiche que les variables avec des na
            print(missing_data[(percent > 0)], '\n')

        print('** Les valeures manquantes **')
        Missing_values(self.dataframe)

        print('** Type des colonne présente dans le dataframe **')
        print(self.dataframe.dtypes.value_counts())







