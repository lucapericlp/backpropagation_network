import pandas as pd
import numpy as np
from NeuralNetwork import NeuralNetwork
from Visualiser import Visualiser

lr = 0.01
epochs = 2500

visualiser = Visualiser(size=111)
df = pd.read_csv('data/4classes.csv',header=None)
# df = pd.read_csv('data/3classes.csv',header=None)
# df = pd.read_csv('data/2classes.csv',header=None)
inputColumns = len(df.columns) - 1
df_input = df.iloc[:,:inputColumns]
df_classes = df.iloc[:,-1]

X = df_input.values.astype(float)
classes_original = df_classes.values.astype(float)
classes_temp = np.array([classes_original])
t = classes_temp.T

# ts = test set
df_ts = pd.read_csv('data/test_set.csv',header=None)
df_ts_input = df_ts.iloc[:,:inputColumns]
X_ts = df_ts_input.values.astype(float)
df_unique_classes = df_classes.unique()

NN = NeuralNetwork(inputSize=inputColumns,outputSize=1,hiddenSize=inputColumns)
NN.train(epochs,X,t,lr,verbose=False)
index_classes = NN.infer(X_ts,df_unique_classes)

# Following was used to output the results into a 3D scatter plot and has intentionally hardcoded values.
# Note: This is only a suitable approach if your dataset has 3 columns, otherwise use a commentblock.

for index,some_class in enumerate(index_classes):
	X,Y,Z = X_ts[index][0],X_ts[index][1],X_ts[index][2]
	colour = [some_class[1],0.5,0.75]
	visualiser.add(np.reshape(X,-1),np.reshape(Y,-1),np.reshape(Z,-1),colour,'o')
	print("{} & {} & {} & {} & {}".format(X,Y,Z,some_class[1],colour))

visualiser.show()