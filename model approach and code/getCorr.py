import pandas as pd
from scipy.stats.stats import pearsonr
import numpy
tp = pd.read_csv('train_v2.csv', iterator=True, chunksize=1000)
df = pd.concat(tp, ignore_index=True)

A = []
B = []

##### Example #########
for i in range(105470):
        A.append(df.values[i][2])
        B.append(df.values[i][3])

print pearsonr(A,B) #(corr,pvalue)
#compute the correlation of f2 and f3

#later on we find corr of f271 and f274 >.99
#and corr of f527,f528 >.99