import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("adult.csv")
df['income'] = df['income'].replace(' <=50K', 0)
df['income'] = df['income'].replace(' >50K', 1)
df = df.dropna()

X = df.drop(columns=['income'])
y = df["income"]
X = pd.get_dummies(X,drop_first = True)
y = pd.get_dummies(y,drop_first = True)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler =StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(x_train,y_train)
y_pred = log_reg.predict(x_test)

tree = DecisionTreeClassifier()

tree.fit(x_train,y_train)

y_tree = tree.predict(x_test)

def performance(model, y,y_pred):
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"--- {model} ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}\n")

performance("Logistic Regression",y_test,y_pred)

performance("Decsison Tree Classifer",y_test,y_tree)
