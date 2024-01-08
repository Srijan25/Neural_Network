import numpy as np
import pandas as pd
A = np.matrix([[1,2,3,4],
[5,5,6,7],
[1,4,2,3],
[5,3,2,1],
[8,1,2,2]])
df = pd.DataFrame(A,columns= ['f1', 'f2','£3','£4'])
df_std = (df - df.mean()) / (df. std())
n_components = 2
from sklearn.decomposition import PCA
pca = PCA(n_components=n_components)
principalComponents = pca. fit_transform(df_std)
principalDf = pd.DataFrame(data=principalComponents, columns=['nf'+str(i+1) for i in range(n_components)])
print(principalDf)
