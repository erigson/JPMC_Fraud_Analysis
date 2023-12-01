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
    def cluster_summary(df, target):    
        # Assuming you have a DataFrame 'df' with 'ClusterLabel' and 'ActualLabel'
        cluster= df[df['ClusterLabel'] == f'cluster{str(target)}']

        # Count the number of fraud and non-fraud cases in cluster3
        fraud_count = cluster['ActualLabel'].value_counts()
        print(f'{fraud_count}\n')


        # Calculate proportions
        # Assuming 'is_fraud' is your fraud label in 'ActualLabel'
        fraud_cases = df[df['ActualLabel'] == 1]
        percentage_of_fraud_in_cluster =  100 * ((cluster['ActualLabel'] == 1).sum() / len(fraud_cases))
        print(f"Percentage of total fraud cases in cluster{str(target)}: {percentage_of_fraud_in_cluster:.2f}%")
        print(f"ratio of fraud cases to non fraud cases in cluster{str(target)}:{len(fraud_cases)}" )
    
    def gmm():
        gmm = GaussianMixture(n_components=3, random_state=0)  # Adjust n_components as needed
        gmm.fit(X_scaled)

        clusters = gmm.predict(X_scaled)

        weights = list(gmm.weights_)
        cluster_labels = []
        for i in range(1, len(weights)+1):
            cluster_labels.append(f'cluster{i}')
        predicted_labels = [cluster_labels[cluster] for cluster in clusters]

        # Create a DataFrame with actual and predicted labels
        cluster_df = pd.DataFrame({'ActualLabel': y_scaled, 'ClusterLabel': predicted_labels})

        cluster_fraud_distribution = cluster_df.groupby('ClusterLabel')['ActualLabel'].value_counts().unstack()

        cluster_fraud_distribution.plot(kind='bar', stacked=True)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Cases')
        plt.title('Distribution of Fraud Cases in Each Cluster')
        plt.show()
