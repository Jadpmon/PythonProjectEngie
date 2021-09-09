import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

class ExploreData:
    _dataframe = None
    _dataframe_path = None

    def __init__(self, dataframe=None, dataframe_path=None):
        if dataframe is not None:
            self._dataframe = dataframe
        if dataframe_path is not None:
            self._dataframe_path: dataframe_path
            self._dataframe = pd.read_csv(self.dataframe_path)

    def info_data(self):
        print( ' ** format du dataframe (ligne,colonne) = {} ** \n'.format(self.dataframe.shape))
        print('** description du data frame **')
        print(self._dataframe.describe())

        def _Missing_values(data):
            total = data.isnull().sum().sort_values(ascending=False)
            percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
            missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Pourcentage'])
            # Affiche que les variables avec des na
            print(missing_data[(percent > 0)], '\n')

        print('** Les valeures manquantes **')
        _Missing_values(self._dataframe)

        print('** Type des colonne présente dans le dataframe **')
        print(self._dataframe.dtypes.value_counts())

    def visualisation_distrib_quanti(self):

        print(' ** DISTRIBUTION DES VARIABLE ** ')

        for col in self._dataframe.select_dtypes([float, int]).columns:
            plt.figure()
            sns.distplot(self._dataframe[col]);
            plt.title(col)


    def visualisation_count_category(self):

        print(' ** COUNT DES VARIABLE CAT ** ')

        for col in self._dataframe.select_dtypes(object).columns:
            plt.figure()
            sns.countplot(x = col, data = self._dataframe, order = self._dataframe[col].value_counts().index);
            plt.title(col)

    def visualisation_corr_matrix(self):
        plt.figure(figsize=(10, 10))
        CorrelationMatrix = self._dataframe.corr()
        sns.heatmap(CorrelationMatrix)



