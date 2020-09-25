# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:15:10 2020

@author: kmruehl
"""

# #  Select all June data, find mean of all June data
data[data.index.month == 6]
data[data.index.month == 6].mean()
len(data[data.index.month == 6])

# plt.boxplot(data[data.index.month==1].Hs)


# plt.boxplot(data[data.index.month==[imonth for imonth in range(len(data.index.month.unique()))]].Hs)

# box_data[1].boxplot()
# box_data[2].boxplot()

# plt.boxplot(data[data.index.month==1].Hs)
# plt.boxplot(data[data.index.month==2].Hs)

# plt.boxplot(box_data[0].Hs)

Jan = box_data[0].Hs
Feb = box_data[1].Hs
Mar = box_data[2].Hs
Apr = box_data[3].Hs
May = box_data[4].Hs
Jun = box_data[6].Hs
Jul = box_data[6].Hs
Aug = box_data[7].Hs
Sep = box_data[8].Hs
Oct = box_data[9].Hs
Nov = box_data[10].Hs
Dec = box_data[11].Hs
plt.boxplot([Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct])

# Jan = box_data[0].Hs
# Feb = box_data[1].Hs
# Mar = box_data[2].Hs
# Apr = box_data[3].Hs
# May = box_data[4].Hs
# Jun = box_data[6].Hs
# Jul = box_data[6].Hs
# Aug = box_data[7].Hs
# Sep = box_data[8].Hs
# Oct = box_data[9].Hs
# Nov = box_data[10].Hs
# Dec = box_data[11].Hs
# # months = [Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct]
# # plt.boxplot(months)
# # plt.boxplot([Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct])

