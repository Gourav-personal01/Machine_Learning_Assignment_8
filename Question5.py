# Q5. Load the wine quality data set and perform exploratory data analysis (EDA) to identify the distribution
# of each feature. Which feature(s) exhibit non-normality, and what transformations could be applied to
# these features to improve normality?

import pandas as pd
import seaborn as sns
df = pd.read_excel('winequality-red.xlsx')

print(df.head())

for i in df.columns:
    sns.histplot(df[i],kde=True)

# residual sugar provides the non-normality functionality and we can use the below transformation to use this as normal distribution: 

# Logarithmic transformation (e.g., np.log)
# Square root transformation (e.g., np.sqrt)
# Box-Cox transformation