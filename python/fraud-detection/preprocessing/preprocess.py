from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline 
import pandas as pd
import numpy as np

df = pd.read_csv("Transactions dataset/creditcard.csv")

# 1. Scale amount, time
# Use MinMax scaling to keep the distribution of points the same, only scale data
# Time and Amount scaled between 0 and 1
scaler = MinMaxScaler()
df[["Amount", "Time"]] = pd.DataFrame(scaler.fit_transform(df[["Amount","Time"]].values), columns=["Amount","Time"], index=df.index)


# Distribute unbalanced dataset
"""refer to 
    https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data#:~:text=An%20effective%20way%20to%20handle,of%20the%20majority%20class%20examples.
    https://www.kaggle.com/code/ramanchandra/handling-imbalanced-datasets
"""

""" Upsampling method (basic)
# seperate cases
not_fraud_df = df[df["Class"] == 0]
fraud_df = df[df["Class"] == 1]

# upsample fraud to match number of not_fraud examples
not_fraud_upsampled = resample(fraud_df, 
                               replace=True,
                               n_samples=len(not_fraud_df),
                               random_state=42) # for testing
# merge resampled fraud data
df = pd.concat([not_fraud_upsampled, not_fraud_df])
"""

""" SMOTE upsampling
Using k-nearest neighbour algorithm to randomly select a random nearest neighbour
    https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/

oversamp = SMOTE(sampling_strategy=0.2, random_state=123) # seed for testing
undersamp = RandomUnderSampler(sampling_strategy=0.5)
# seperate data from class
# upsample fraud (1) with multiple instances of itself
steps = [('o', oversamp), ('u', undersamp)]
pipeline = Pipeline(steps=steps)
x, y = pipeline.fit_resample(df.drop("Class", axis=1), df["Class"])
sampled_df = pd.concat([pd.DataFrame(x),pd.DataFrame(y)])
"""
smtok = SMOTEENN(random_state=123) # seed for testing
# seperate data from clas
x, y = smtok.fit_resample(df.drop("Class", axis=1), df["Class"])
sampled_df = pd.concat([pd.DataFrame(x),pd.DataFrame(y)])

print(sampled_df["Class"].value_counts())
