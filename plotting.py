import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

epex_price = pd.read_csv(".\epex_day_ahead_price.csv")

y1 = range(0,13204)
y2 = range(13204,30772)
y3 = range(30772,47860)

year_1 = pd.read_csv(".\epex_day_ahead_price.csv", skiprows = [x for x in y2 or y3])
year_2 = pd.read_csv(".\epex_day_ahead_price.csv", skiprows = [x for x in y1 or y3])
year_3 = pd.read_csv(".\epex_day_ahead_price.csv", skiprows = [x for x in y1 or y2])

#epex_price.plot(y = 'apx_da_hourly',kind="line")
#plt.show()

year_1.plot.hist(bins=1000)
plt.show()

"""
ax = year_1.plot(kind = "line")
year_2.plot(kind="line",ax=ax)
year_3.plot(kind="line",ax=ax)
plt.show()"""

"""
plt.plot(year_1["Price"])
plt.plot(year_2["Price"])
plt.plot(year_3["Price"])
plt.show()"""
