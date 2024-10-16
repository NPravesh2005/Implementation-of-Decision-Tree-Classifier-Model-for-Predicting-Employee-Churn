# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import pandas module and import the required data set
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: PRAVESH N
RegisterNumber:  212223230154
*/
```

```python
# import the pandas library to read the csv file
import pandas as pd
df = pd.read_csv("/content/Employee (1).csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df['left'].value_counts())

# import LabelEncoder to enhance the performance by giving distinct numerical value to categorical value
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# standardizing the salary values into some specific range for enhancing the performance
df['salary'] = le.fit_transform(df['salary'])
# print(df)

x = df[['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
#x = df.iloc[:,[0,1,2,3,4,5,7,9]]
print(x)
y = df['left']
print(y)

#import train_test_split to split the dataset into training data and testing data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 30)
# print("X - Train : \n",x_train)
# print("X - Test : \n",x_test)
# print("Y - Train : \n",y_train)
# print("Y - Test : \n",y_test)

#import the DecisionTreeClassifier to determine the root node and leaf nodes
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
y_pred = dt.predict(x_test)
print("Y Predicted : \n\n",y_pred)

#import metrics to find the performance measure of the machine learning model
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"\nAccuracy : {accuracy * 100:.2f}%")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

```python
df.head()
```
![Screenshot 2024-10-16 101256](https://github.com/user-attachments/assets/8796c937-ea68-4443-994b-6d466aca082f)

```python
df.info()
```
![Screenshot 2024-10-16 101446](https://github.com/user-attachments/assets/27f44ea8-bf93-4604-acf6-4f5174352080)

```python
df.isnull().sum()
```
![Screenshot 2024-10-16 101601](https://github.com/user-attachments/assets/b2711cd3-b198-415c-8899-310f8841a569)

```python
df['left'].value_counts()
```
![Screenshot 2024-10-16 101712](https://github.com/user-attachments/assets/d04a8a74-e059-4ddf-b5be-48d22f502dac)

```python
print(x)
y = df['left']
print(y)
print("Y Predicted : \n\n",y_pred)
print(f"\nAccuracy : {accuracy * 100:.2f}%")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![Screenshot 2024-10-16 101956](https://github.com/user-attachments/assets/362743e6-c6cb-441e-8cd1-6e1b11c535a6)
![Screenshot 2024-10-16 102030](https://github.com/user-attachments/assets/02e0ce08-ca34-4417-b127-827581211d19)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
