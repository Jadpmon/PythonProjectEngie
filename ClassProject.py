import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
        print(' ** format du dataframe (ligne,colonne) = {} ** \n'.format(self._dataframe.shape))
        print('** description du data frame **')
        print(self._dataframe.describe())

        def _missing_values(data):
            total = data.isnull().sum().sort_values(ascending=False)
            percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)
            missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Pourcentage'])
            # Affiche que les variables avec des na
            print(missing_data[(percent > 0)], '\n')

        print('** Les valeures manquantes **')
        _missing_values(self._dataframe)

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
            sns.countplot(x=col, data=self._dataframe, order=self._dataframe[col].value_counts().index);
            plt.title(col)

    def visualisation_corr_matrix(self):
        print('** Matrice de correlation **')
        plt.figure(figsize=(10, 10))
        correlationmatrix = self._dataframe.corr()
        sns.heatmap(correlationmatrix)

    def columns_to_drop(self, columns):
        print('** Suppression des colonne : ', columns, ' **')
        self._dataframe.drop(columns, axis=1, inplace=True)

    def process_cat(self):
        for c in self._dataframe.select_dtypes(object).columns:
            self._dataframe[c] = LabelEncoder().fit_transform(self._dataframe[c].astype(str))


class Model:
    _X_train = None
    _X_test = None
    _y_train = None
    _y_test = None
    _regression_models = {}
    _classification_models = {}
    # 0 pour regression 1 pour classfication
    _regress_or_classfication = 0
    _target = None

    def __init__(self, dataframe, type_model=0):
        self._dataframe = dataframe
        self._regress_or_classfication = type_model

    def train_test_split(self, col_name_target, test_size=0.2):
        _stratify = None
        self._target = self._dataframe[col_name_target]
        _tmp_data = self._dataframe.drop(col_name_target, axis=1)
        if self._regress_or_classfication == 1:
            _stratify = self._target
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(_tmp_data, self._target,
                                                                                    test_size=test_size,
                                                                                    stratify=_stratify)

    def init_models(self):
        # initialisation des modeles de regression
        if self._regress_or_classfication == 0:
            self._regression_models['gbc'] = {'model': GradientBoostingRegressor(), 'name': 'GradientBoostingRegressor'}
            self._regression_models['rf'] = {'model': RandomForestRegressor(), 'name': 'RandomForestRegressor'}
            self._regression_models['tree'] = {'model': DecisionTreeRegressor(), 'name': 'DecisionTreeRegressor'}
            self._regression_models['svc'] = {'model': SVR(), 'name': 'SVR'}
            self._regression_models['knn'] = {'model': KNeighborsRegressor(), 'name': 'KNeighborsRegressor'}
            self._regression_models['lr'] = {'model': LinearRegression(), 'name': 'LinearRegression'}
        # initialisations des modeles de classifications
        else:
            pass

    def evaluation(self, metric=None):
        if self._regress_or_classfication == 0:
            for model in self._regression_models:
                self._regression_models[model]['score'] = cross_val_score(self._regression_models[model]['model'],
                                                                          self._X_train, self._y_train, cv=3,
                                                                          scoring=metric)
                print(self._regression_models[model]['name'] + ": %0.4f (+/- %0.4f)" % (
                self._regression_models[model]['score'].mean(),
                self._regression_models[model]['score'].std() * 2))
        else:
            pass
