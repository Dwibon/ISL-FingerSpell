import pandas as pd

df1 = pd.read_csv('/Users/dwibon/Desktop/ISL-FingerSpell/dataset/person1_dataset.csv')
df2 = pd.read_csv('/Users/dwibon/Desktop/ISL-FingerSpell/dataset/person2_dataset.csv')

combined = pd.concat([df1, df2], ignore_index= True)
combined = combined.dropna()
combined.to_csv("training_data.csv", index = False)

print(f"Combined Shape: {combined.shape}")
print(combined['label'].value_counts())
