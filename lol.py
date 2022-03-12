import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

# relaxes display limits
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# read csv file
data = pd.read_csv('epex_day_ahead_price.csv')

# set data column to be a data frame index
data.index = pd.to_datetime(data['timestamp'], format = '%Y-%m-%d')
del data['timestamp']

# format visualisation using Seaborn
sns.set()
# plt.ylabel('apx_da_hourly')
# plt.xlabel('timestamp')
# plt.xticks(rotation=45)
# plt.plot(data.index, data['apx_da_hourly'])
# plt.show()

# Splitting Data for Training & Testing
frog = "2021-09-01"
train = data[data.index < frog]
test = data[data.index > frog]

plt.plot(train, color='black')
plt.plot(test, color='red')
plt.ylabel('apx_da_hourly')
plt.xlabel('timestamp')
plt.xticks(rotation=45)
plt.title('Train/TEst split for apx data')
plt.show()