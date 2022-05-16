import pandas as pd
import numpy as np

dataset=pd.read_csv("dt.csv")
print("\n Dataset is --> \n", dataset)
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,5]
 
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

X=X.apply(le.fit_transform)
print("\n X --> \n ",X)
features=['AGE','INCOME','GENDER','MARTIAL STATUS']
labels=['YES','NO']

from sklearn.tree import DecisionTreeClassifier
regressor=DecisionTreeClassifier()
regressor.fit(X.iloc[:,1:5],y)

X_in=np.array([1,1,0,0])
y_pred=regressor.predict([ X_in ])
print("\n Prediction = ", y_pred)

#from sklearn.externals.six import StringIO
from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz 
import pydotplus

dot_data=StringIO()

export_graphviz(regressor,out_file=dot_data,filled=True,rounded=True,special_characters=True,impurity=None,feature_names=features, class_names=labels)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('treeOP.png')
 
