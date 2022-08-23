#standard ds libraries
import pandas as pd
import numpy as np

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
    df['number_relationships'] = df['dependents'] + df['partner']

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