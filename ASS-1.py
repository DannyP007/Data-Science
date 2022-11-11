# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 22:15:54 2022

@author: mauli
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read Csv File 

spend = pd.read_csv("Ass-1.csv")
print(spend)


#Adding labels to pie graph
name = ["Uk", "Germany", "Sweden", "Spain", "France"]

# Plotting pie chart with values
plt.figure()
#plt.pie(spend["Percentages"], labels=name, autopct='%1.1f%%') # 'autopct' is used for adding percentage to portions
plt.pie(spend["Euro/person"], labels=name, autopct='%1.1f%%')
plt.title("The Average spending shopping on the internet per person")
plt.show()

print('------------------------------------------------------')

# BAR GRAPH

#READ CSV FILE

eth = pd.read_csv("ETH-USD.csv")
print(eth)

plt.figure(figsize=(13,10))

plt.bar(eth["Date"], eth["Open"], width=0.4)

# General Matters
plt.title("Price of ETH")
plt.xlabel("Year")
plt.ylabel("Price in Dollars")
plt.show()


print('------------------------------------------------------')

# LINE GRAPH

btc = pd.read_csv("BTC.csv")
print(btc)
# for plot line 
plt.figure(figsize=(15,10))
plt.plot(btc["Date"],btc["Open"],label="open price")
plt.plot(btc["Date"],btc["Volume (BTC)"],label="Volume")

plt.title("BTC PRICE BITWEEN 2011 TO 2021") 
plt.xlabel("Year") # label for x and y Axis
plt.ylabel("Price in USD Dollars")
plt.legend()
plt.show()
