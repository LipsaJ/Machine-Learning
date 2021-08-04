
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
import numpy as np

blues = pd.read_csv("data/blues_hand.csv")

blues['brthYr1'] = np.floor(blues['brthYr']/50) 
print(blues.head())

X = blues[['region','handPost','thumbSty']]
X1 = blues[['handPost','thumbSty']]
y = blues['brthYr1']
y1 = blues['brthYr1']
a = []
b = []




for i in range(0,11):
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1, test_size=0.3)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train,y_train)
    predictions=dtree.predict(X_test)
    dtree.fit(X1_train,y1_train)
    predictions1=dtree.predict(X1_test)
    a.append(accuracy_score(y1_test,predictions1))
    b.append(accuracy_score(y_test,predictions))
#print(confusion_matrix(y_test,predictions))

print(min(a),max(a), sum(a)/len(a))
print(min(b),max(b), sum(b)/len(b))

text_representation = tree.export_text(dtree)
#print(text_representation)





# =============================================================================
# X = TipJoke[['Ad', 'Joke', 'None']]
# y = TipJoke['Tip']
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# 
# dtree = DecisionTreeClassifier()
# dtree.fit(X_train, y_train)
# predictions = dtree.predict(X_test)
# 
# print(accuracy_score(y_test, predictions))
# print(confusion_matrix(y_test, predictions))
# text_representation = tree.export_text(dtree)
# print(text_representation)
# 
# # Install it using 'pip install graphviz'
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)

c =[]
d = []
for i in range(0,11):
    rfc.fit(X1_train, y1_train.values.ravel())
    predictions1 = rfc.predict(X1_test)
    c.append(accuracy_score(y1_test, predictions1))

    rfc.fit(X_train, y_train.values.ravel())
    predictions = rfc.predict(X_test)
    d.append(accuracy_score(y_test, predictions))
    
print(min(c),max(c), sum(c)/len(c))
print(min(d),max(d), sum(d)/len(d))
    
#print(confusion_matrix(y_test, predictions))