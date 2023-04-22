# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1
Read the given Data

STEP 2
Clean the Data Set using Data Cleaning Process

STEP 3
Apply Feature Transformation techniques to all the features of the data set

STEP 4
Save the data to the file

# CODE
/*
~~~
Name:PRATHIKSHA V
Reg no:212220220028
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
df = pd.read_csv("/content/Data_to_Transform.csv")
print(df)
df.head()
df.isnull().sum()
df.info()
df.describe()
df1 = df.copy()
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.HighlyNegativeSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModeratePositiveSkew,fit=True,line='45')
plt.show()
sm.qqplot(df1.ModerateNegativeSkew,fit=True,line='45')
plt.show()
df1['HighlyPositiveSkew'] = np.log(df1.HighlyPositiveSkew)
sm.qqplot(df1.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df2 = df.copy()
df2['HighlyPositiveSkew'] = 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df3 = df.copy()
df3['HighlyPositiveSkew'] = df3.HighlyPositiveSkew**(1/1.2)
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')
plt.show()
df4 = df.copy()
df4['ModeratePositiveSkew_1'],parameters =stats.yeojohnson(df4.ModeratePositiveSkew)
sm.qqplot(df4.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()
from sklearn.preprocessing import PowerTransformer 
trans = PowerTransformer("yeo-johnson")
df5 = df.copy()
df5['ModerateNegativeSkew_1'] = pd.DataFrame(trans.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_1'],line='45')
plt.show()
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df5['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df5[['ModerateNegativeSkew']]))
sm.qqplot(df5['ModerateNegativeSkew_2'],line='45')
plt.show()
~~~
*/
# OUPUT
# Feature Transformation
![image](https://user-images.githubusercontent.com/103166779/197209713-daadc844-da11-4cf6-b4db-3abcb9164781.png)

![image](https://user-images.githubusercontent.com/103166779/197209931-f5a1e8ce-0510-4df7-8ea3-c79de7a0918a.png)

![image](https://user-images.githubusercontent.com/103166779/197210050-991e27a7-0a58-4e7b-bb56-0ef64be38715.png)

![image](https://user-images.githubusercontent.com/103166779/197210302-f7e8af9a-8885-48aa-a652-5bfc2cbbfa9c.png)

![image](https://user-images.githubusercontent.com/103166779/197210503-89a66735-6965-471a-94a2-aa4f7b59f132.png)

![image](https://user-images.githubusercontent.com/103166779/197210740-c79e6a46-4ba0-4990-8883-1f7272292a7a.png)

![image](https://user-images.githubusercontent.com/103166779/197211065-baf2f24c-bf69-46df-867d-460ffb6139a3.png)

![image](https://user-images.githubusercontent.com/103166779/197211906-71b64a7a-cdc7-4c23-8530-f2bb0387c34b.png)

![image](https://user-images.githubusercontent.com/103166779/197212170-a670cee1-025e-4649-9584-88a7e4e28af5.png)

![image](https://user-images.githubusercontent.com/103166779/197212326-63f76916-efc5-4371-abce-91af068bbc48.png)

![image](https://user-images.githubusercontent.com/103166779/197212506-f112219f-67ff-4b77-b659-bd7924d52ff6.png)

![image](https://user-images.githubusercontent.com/103166779/197212765-95aef12a-fa7f-41fc-9514-b195c2d6a171.png)

![image](https://user-images.githubusercontent.com/103166779/197212931-885da1ad-b499-4302-979e-f51c381f545f.png)

# RESULT
Thus the Feature Transformation for the given datasets had been executed successfully

