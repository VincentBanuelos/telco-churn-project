#standard ds libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import splitting functions
from sklearn.model_selection import train_test_split

def prep_telco(df):
    '''
    Prepares, tidys, and cleans the data 
    so that it is ready for exploration and analysis 
    '''

    df['total_charges'] = pd.to_numeric(df['total_charges'], errors='coerce')
    df['tenure'].astype(float)
    df['is_male'] = df.gender.map({'Male': 1, 'Female': 0})
    df.drop(columns='gender', inplace=True)
    df['multiple_lines'] = df.multiple_lines.map({'Yes': 1, 'No': 0, 'No phone service' : 0})

    YN_features = ['churn', "paperless_billing","phone_service","dependents","partner"]
    for i in YN_features:
        df[i] = df[i].map({'Yes': 1, 'No': 0})
    
    df['num_addons'] = (df[['online_security', \
                        'online_backup', \
                        'device_protection', \
                        'tech_support', \
                        'streaming_tv', \
                        'streaming_movies', \
                        'contract_type', \
                        'internet_service_type', \
                        'payment_type']] == 'Yes').sum(axis=1)
    
    df['number_relationships'] = df['dependents'] + df['partner']

    dummy_df = pd.get_dummies(df[['multiple_lines', \
                              'online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type'
                            ]],
                              drop_first=True)
    dummy_df = pd.get_dummies(df[['online_security', \
                              'online_backup', \
                              'device_protection', \
                              'tech_support', \
                              'streaming_tv', \
                              'streaming_movies', \
                              'contract_type', \
                              'internet_service_type', \
                              'payment_type'
                            ]],
                              drop_first=True)
    df = pd.concat( [df, dummy_df], axis=1 )

    return df



def my_train_test_split(df, target):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    '''
    train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])
    train, validate = train_test_split(train, test_size=0.3, random_state=123, stratify=train[target])
    
    return train, validate, test



plots = {1 : [111], 2: [121, 122], 3: [131, 132, 133], 4: [221, 222, 223, 224], 5: [231, 232, 233, 234, 235], 6: [231, 232, 233, 234, 235, 236]}

def countplot(x, y, train):
    '''
    This function will create lineplots that show visually how a feature
    relates to churn.
    '''
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    
    plt.figure(figsize=(7*columns, 7*rows))
    
    for i, j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.countplot(x=j, hue=x, data=train, alpha=0.8, linewidth=0.4, edgecolor='black')
        ax.set_title(j)
        
    return plt.show()
