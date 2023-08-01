import pandas as pd
import numpy as np
'''
df = pd.DataFrame(columns = ["fcn_acc", "fcn_time", "Deepmask_acc", "Deepmask_time", "total_time", "", "detectron_acc", "detectron_time"])

df.to_csv("acc.csv", index=False)
'''

df = pd.read_csv("../test/acc.csv")
test = [0.1,1.1,2.1,3.1,4.1,6.1]
df.at[:, "fcn_acc"] = test[:]
df.at[:, "test"] = test[:]
df.at[:, "fcn_accuracy"] = test[:]

print(df)
df.to_csv("test2.csv", index=False)

