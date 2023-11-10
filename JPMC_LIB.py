import pandas as pd
import numpy as np
import os 
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans

from sklearn.model_selection import StratifiedKFold, GridSearchCV, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import category_encoders as ce

class JPMC_LIB():
    
    @staticmethod
    def plot_gender_fraud_relationship(data, gender_column='gender', fraud_column='is_fraud'):
        ax = sns.histplot(x=gender_column, data=data, hue=fraud_column, stat='percent', multiple='dodge', common_norm=False)
        ax.set_ylabel('Percentage')
        ax.set_xlabel('Credit Card Holder Gender')
        plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])
        plt.show()  # This line ensures the plot is displayed when the function is called

    @staticmethod
    def plot_transaction_amount_distribution(df, amount_column='amt', fraud_column='is_fraud'):
        ax = sns.histplot(x=amount_column, data=df[df[amount_column] <= 1000], hue=fraud_column, stat='percent', multiple='dodge', common_norm=False, bins=25)
        ax.set_ylabel('Percentage in Each Type')
        ax.set_xlabel('Transaction Amount in USD')
        plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])
        plt.show()

    @staticmethod
    def plot_age_distribution(df, age_column='age', fraud_column='is_fraud'):
        ax = sns.kdeplot(x=age_column, data=df, hue=fraud_column, common_norm=False)
        ax.set_xlabel('Credit Card Holder Age')
        ax.set_ylabel('Density')
        plt.xticks(np.arange(0, 110, 5))
        plt.title('Age Distribution in Fraudulent vs Non-Fraudulent Transactions')
        plt.legend(title='Type', labels=['Fraud', 'Not Fraud'])
        plt.show()

    @staticmethod
    def plot_fraud_percentage_bytime(df, title, xlabel, fraud_column='Fraud', total_column='Total'):
        df['Fraud_Percentage'] = (df[fraud_column] / df[total_column]) * 100
        ax = df['Fraud_Percentage'].plot(kind='bar', figsize=(12, 6), color='#ff7f0e')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('Fraud Percentage (%)')
        plt.show()

    @staticmethod
    def plot_fraud_rate_by_category(df, category_column='category', fraud_column='is_fraud'):
        fraud_rate_category = df.groupby(category_column)[fraud_column].mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(y=fraud_rate_category.index, x=fraud_rate_category.values, ax=ax, orient='h', color='#ff7f0e')
        ax.set_title('Fraud Rate by Spending Category')
        ax.set_xlabel('Fraud Rate')
        ax.set_ylabel('Spending Category')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def resampling_evaluation(X_train, y_train, X_test, y_test, classifier, sampling_technique='undersample', rate=0.1):
        if sampling_technique == 'undersample':
            sampler = RandomUnderSampler(sampling_strategy=rate)
        elif sampling_technique == 'oversample':
            sampler = RandomOverSampler(sampling_strategy=rate)
        elif sampling_technique == 'smote':
            sampler = SMOTE(sampling_strategy=rate)
        else:
            raise ValueError("Invalid sampling technique. Choose from 'undersample', 'oversample', or 'smote'.")

        # Perform sampling
        X_sampled, y_sampled = sampler.fit_resample(X_train, y_train)
        
        # Fit the classifier
        classifier.fit(X_sampled, y_sampled)
        
        # Predict on the test set
        sampled_pred = classifier.predict(X_test)
        
        # Calculate evaluation metrics
        sampled_metrics = {
            'recall': recall_score(y_test, sampled_pred),
            'precision': precision_score(y_test, sampled_pred),
            'f1': f1_score(y_test, sampled_pred),
            'roc_auc': roc_auc_score(y_test, sampled_pred)
        }
        return sampled_metrics
    

    @staticmethod
    def kmean_hyper_param_tuning(data):
        """
        Hyper parameter tuning to select the best from all the parameters on the basis of silhouette_score.

        :param data: dimensionality reduced data after applying PCA
        :return: best number of clusters for the model (used for KMeans n_clusters)
        """
        # candidate values for our number of cluster
        parameters = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]

        # instantiating ParameterGrid, pass number of clusters as input
        parameter_grid = ParameterGrid({'n_clusters': parameters})

        best_score = -1
        kmeans_model = KMeans()     # instantiating KMeans model
        silhouette_scores = []

        # evaluation based on silhouette_score
        for p in parameter_grid:
            kmeans_model.set_params(**p)    # set current hyper parameter
            kmeans_model.fit(data)          # fit model on wine dataset, this will find clusters based on parameter p

            ss = silhouette_scores(data, kmeans_model.labels_)   # calculate silhouette_score
            silhouette_scores += [ss]       # store all the scores

            print('Parameter:', p, 'Score', ss)

            # check p which has the best score
            if ss > best_score:
                best_score = ss
                best_grid = p

        # plotting silhouette score
        plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
        plt.xticks(range(len(silhouette_scores)), list(parameters))
        plt.title('Silhouette Score', fontweight='bold')
        plt.xlabel('Number of Clusters')
        plt.show()

        return best_grid['n_clusters']    