import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\trainDelayPrediction\data\train delay data.csv')
df.head()
df.info()
df.describe()

df.isnull().sum()

sns.histplot(df['Historical Delay (min)'], bins=30, kde=True)
plt.title('Distribution of Train Delays')