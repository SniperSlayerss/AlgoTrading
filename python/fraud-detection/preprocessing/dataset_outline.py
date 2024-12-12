import pandas

df = pandas.read_csv("Transactions dataset/creditcard.csv")
sep = "-"*30

print(sep)
# Here we can see each transaction
# V1-V28 are features absracted, redacted for privacy reasons
print("Columns:\n", df.columns)

print(sep)
# All features have went through a PCA transformation (excluding time and amount),
# to implement this the features would have been previously scaled,
# therefore all V features have been scaled
print("Row (example)")
print(df.iloc[0])
""" Class represents:
            0 - not fraud
            1 - fraud
"""
print(sep)
# Amount values small, mean of 88 USD
# Imbalanced dataset, mean of class is 0.0073, most transactions are not fraud
# Need to scale and account for imbalance
print(df.mean(axis=0))

print(sep)
# Number of fraud transactions
fraud_sum = df['Class'].value_counts()
print("Fraud:", fraud_sum[0], fraud_sum[0]/len(df)) # 99.8%
print("No Fraud:", fraud_sum[1], fraud_sum[1]/len(df)) # 0.17%

