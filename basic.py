import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


    #Load the data
df = pd.read_csv('data.csv')

    #check the first 10 lines of the imported data
print(df.head(10))

    #to check the rows and coloums of the data set
print(df.shape)

    #Count the empty (NaN, NAN, na) values in each column
print(df.isna().sum())
    #Drop the column with all missing values (na, NAN, NaN)
    #NOTE: This drops the column Unnamed
df = df.dropna(axis=1)

    #Get the new count of the number of rows and cols
df.shape

    #Get a count of the number of 'M' & 'B' cells
count_M_and_B = df['diagnosis'].value_counts()

    #Visualize this count
sns.countplot(df['diagnosis'],label="Count")


    #Encoding categorical data values (

labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))
sns.pairplot(df, hue="diagnosis")

df.corr()
plt.show()

    #Splitting the Dataset into test and train

X = df.iloc[:, 2:31].values
Y = df.iloc[:, 1].values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

    #Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def models(X_train, Y_train):
        # Using Logistic Regression

    logistics = LogisticRegression(random_state=0)
    logistics.fit(X_train, Y_train)

        # Using DecisionTreeClassifier

    Dtree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    Dtree.fit(X_train, Y_train)

        # Using RandomForestClassifier

    RFC = RandomForestClassifier(n_estimators=45, criterion='entropy', random_state=0)
    RFC.fit(X_train, Y_train)

        # print model accuracy on the training data.
    print('[0]Logistic Regression Training Accuracy:', logistics.score(X_train, Y_train))
    print('[1]Decision Tree Classifier Training Accuracy:', Dtree.score(X_train, Y_train))
    print('[2]Random Forest Classifier Training Accuracy:', RFC.score(X_train, Y_train))

    return logistics, Dtree, RFC

    #running the model with the preeprocessed data
model = models(X_train,Y_train)


    # Checking Accuracy with Confusion Matirx
for i in range(len(model)):
    Cmartix = confusion_matrix(Y_test, model[i].predict(X_test))

    TN = Cmartix[0][0]
    TP = Cmartix[1][1]
    FN = Cmartix[1][0]
    FP = Cmartix[0][1]


    print('Model[{}] Testing Accuracy = "{}!"'.format(i, (TP + TN) / (TP + TN + FN + FP)))
    print(Cmartix)
    print("===================")