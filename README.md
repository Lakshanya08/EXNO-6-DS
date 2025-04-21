# EXNO-6-DS-DATA VISUALIZATION USING SEABORN LIBRARY

# Aim:
  To Perform Data Visualization using seaborn python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:
Name:Lakshanya.N
Reg No: 212224230136
```
import seaborn as sns
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[3,6,2,7,1]
sns.lineplot(x=x,y=y)
```
![image](https://github.com/user-attachments/assets/bf2c6d97-2cd6-43d7-85cf-d9499330ea31)

```
df=sns.load_dataset("tips")
df
```
![image](https://github.com/user-attachments/assets/666102f3-3ecc-43d6-b6d5-0d0287e35adf)

```
sns.lineplot(x="total_bill",y="tip",data=df,hue="sex",linestyle='solid',legend='full')
```
![image](https://github.com/user-attachments/assets/433a98e4-21de-4e62-953b-72b49217cb9b)

```
x=[1,2,3,4,5]
y1=[3,5,2,6,1]
y2=[1,6,4,3,8]
y3=[5,2,7,1,4]
sns.lineplot(x=x,y=y1)
sns.lineplot(x=x,y=y2)
sns.lineplot(x=x,y=y3)
plt.title('Multi-:Line Plot')
plt.xlabel('X-Label')
plt.ylabel('Y-Label')
```
![image](https://github.com/user-attachments/assets/3b2c5bc7-8cb9-4071-8c99-2af2d492fcbd)

```
tips=sns.load_dataset('tips')
avg_total_bill=tips.groupby('day')['total_bill'].mean()
avg_tip=tips.groupby('day')['tip'].mean()
plt.figure(figsize=(8,6))
p1=plt.bar(avg_total_bill.index,avg_total_bill,label='Total Bill')
p2=plt.bar(avg_tip.index,avg_tip,label='Tip')
plt.xlabel('Day of the Week')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Day')
plt.legend()
```
![image](https://github.com/user-attachments/assets/48ad94e7-dfbf-4b53-bb4e-780d94c9cc37)

```
avg_total_bill=tips.groupby('time')['total_bill'].mean()
avg_tip=tips.groupby('time')['tip'].mean()
p1=plt.bar(avg_total_bill.index,avg_total_bill,label='Total Bill',width=0.4) # Added a comma after 0.4
p2=plt.bar(avg_tip.index,avg_tip,bottom=avg_total_bill,label='Tip',width=0.4) # Added a comma after 0.4
plt.xlabel('Time of Day')
plt.ylabel('Amount')
plt.title('Average Total Bill and Tip by Time of Day')
plt.legend()
```
![image](https://github.com/user-attachments/assets/5fbc1e96-a761-4ff2-bacf-5a9ebd58438d)

```
years = range(2000, 2012)
apples = [0.895, 0.91, 0.919, 0.926, 0.929, 0.931, 0.934, 0.936, 0.937, 0.9375, 0.9372, 0] # Added comma after 0
oranges = [0.962, 0.941, 0.930, 0.923, 0.918, 0.908, 0.907, 0.904, 0.901, 0.898, 0.9, 0.8] # Added comma after 0.8
plt.bar(years, apples)
plt.bar(years, oranges, bottom=apples)
```
![image](https://github.com/user-attachments/assets/7255f89e-8e32-455d-bb9d-77e3f9033a43)

```
import seaborn as sns
dt=sns.load_dataset('tips')
sns.barplot(x='day',y='total_bill',hue='sex',data=dt,palette='Set1')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill')
plt.title('Total Bill by Day and Gender')
```
![image](https://github.com/user-attachments/assets/273ec961-66e7-4f09-9e66-8f808405c56a)

```
import pandas as pd
tit=pd.read_csv("/content/titanic_dataset (2).csv")
tit
```
![image](https://github.com/user-attachments/assets/57099e4d-1655-4cc5-9aa6-25d755c7da48)

```
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked',y='Fare',data=tit,palette='rainbow')
plt.title("Fare of Passenger by Embarked Town")
```
![image](https://github.com/user-attachments/assets/61e1db62-1edc-4e3e-a748-4a3bd664281c)

```
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked',y='Fare',data=tit,palette='rainbow',hue='Pclass')
plt.title("Fare of Passenger by Embarked Town,Divided by Class")
```
![image](https://github.com/user-attachments/assets/fc9abea7-a9ae-4c5d-83e7-5ee9de10321b)

```
tips=sns.load_dataset('tips')
sns.scatterplot(x='total_bill',y='tip',hue='sex',data=tips)
plt.xlabel('Total Bill')
```
![image](https://github.com/user-attachments/assets/8e01fb22-cbcc-4524-a147-a26b79a885dd)

```
import seaborn as sns
import numpy as np
import pandas as pd
np.random.seed(1)
num_var=np.random.randn(1000)
num_var=pd.Series(num_var,name="Numerical Variable")
num_var
```
![image](https://github.com/user-attachments/assets/b39acadf-0131-4873-9459-59c292fa984a)

```
sns.histplot(data=num_var,kde=True)
```
![image](https://github.com/user-attachments/assets/3a5c24d9-d44c-4cb1-80da-ff9073cec61e)

```
sns.histplot(data=tit,x="Pclass",hue="Survived",kde=True)
```
![image](https://github.com/user-attachments/assets/f035a072-715b-48ec-868e-961ca07f8f01)
```
import matplotlib.pyplot as plt
np.random.seed(0)
marks=np.random.normal(loc=70,scale=10,size=100)
marks
```
![image](https://github.com/user-attachments/assets/2477b159-6cb4-4db2-b636-2563bb68e5d7)

```
sns.histplot(data=marks,bins=10,kde=True,stat='count',cumulative=False) # Added closing parenthesis
plt.xlabel('Marks')
plt.ylabel('Density')
plt.title('Histogram of Student Marks')
```
![image](https://github.com/user-attachments/assets/e27d0556-727b-4cd1-babb-6ab8073eaaa1)

```
tips=sns.load_dataset('tips')
sns.boxplot(x=tips['day'],y=tips['total_bill'],hue=tips['sex'])
```
![image](https://github.com/user-attachments/assets/aedbf689-8ea5-4835-863b-38171dc3760c)

```
sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, linewidth=2, width=0.8,
          whiskerprops={"color": "black", "linestyle": "--", "linewidth": 1.5})

```
![image](https://github.com/user-attachments/assets/4f0e99b0-8eb6-4486-817a-445f6f8afb9f)

```
sns.boxplot(x='Pclass',y='Age',data=tit,palette='rainbow')
plt.title("Age by Passenger Class,Titanic")
```
![image](https://github.com/user-attachments/assets/3f8d6586-e160-41fc-8ae0-e39839fcaf70)

```
sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, linewidth=2, width=0.8) # Added closing parenthesis and provided a value for width
plt.xlabel("Day of the week")
plt.ylabel("Total Bill")
plt.title("Violin Plot of Total Bill by Day and Smoker Status")
```
![image](https://github.com/user-attachments/assets/7f7ee9a8-ce8f-4c00-9b5d-1bab925544df)

```
import seaborn as sns
sns.set(style='whitegrid')
tip=sns.load_dataset('tips')
sns.violinplot(x='day',y='tip',data=tip)
```
![image](https://github.com/user-attachments/assets/cca1622a-aacd-4004-b724-9fe57bad3e75)

```
sns.set(style='whitegrid')
tip=sns.load_dataset('tips')
sns.violinplot(x=tip["total_bill"])
```
![image](https://github.com/user-attachments/assets/98c6450d-e00d-4b43-b675-7a8f1e39dc71)

```
sns.set(style='whitegrid')
tip=sns.load_dataset('tips')
sns.violinplot(x="tip",y="day",data=tip)
```
![image](https://github.com/user-attachments/assets/d63201a1-3c14-43c7-a797-113cbc627b12)

```
sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='fill',linewidth=3) # Added closing parenthesis
```
![image](https://github.com/user-attachments/assets/704a01a1-66f9-484a-8d38-8121ebd8453d)

```
sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='layer',linewidth=3)
```
![image](https://github.com/user-attachments/assets/ba130d00-fd6b-4669-ba13-2d9d23df07c0)

```
sns.kdeplot(data=tips, x='total_bill', hue='time', multiple='stack', linewidth=3)
```
![image](https://github.com/user-attachments/assets/05e8343b-8a5e-4335-ad9a-4c5cd2642a74)

```
data=np.random.randint(low=1,high=100,size=(10,10))
print("The data to be plotted:\n")
print(data)
```
![image](https://github.com/user-attachments/assets/d14ed390-e8fb-488b-b151-2dba134796e3)

```
hm=sns.heatmap(data=data)
```
![image](https://github.com/user-attachments/assets/3e7cb413-53ef-46bd-9370-b104e6546bc4)

```
tips=sns.load_dataset('tips')
numeric_cols=tips.select_dtypes(include=np.number).columns
corr=tips[numeric_cols].corr()
sns.heatmap(corr,annot=True,cmap="plasma",linewidth=0.5)
```
![image](https://github.com/user-attachments/assets/1dcbc0bc-002e-4c23-bb42-8c9bba097e2e)

# Result:
 Data Visualization using seaborn python library for the given datas has been performed
successfully.
