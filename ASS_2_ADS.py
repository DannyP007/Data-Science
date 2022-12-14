# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 23:35:47 2022

@author: mauli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Download data from the World bank data and store the file where python file existing
df_renew = pd.read_csv('Renewable Energy.csv')
print(df_renew)

#drop the columns names from the files
df_renew = df_renew.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)
print(df_renew)

#replace the '..' into None Values
df_renew = df_renew.replace('..',np.NaN)

# acess perticular raw through index location function
df_renew_r = df_renew.iloc[[29,35,40,55,70,81,109,116,259], :]
print(df_renew_r)

# Calling Counrty name column
Cnames = df_renew_r['Country Name']
print(Cnames)

# set the values for x-axis 
X_axis = np.arange(len(Cnames))

# set the figure size for better visualization
plt.figure(figsize=(10,8), dpi=500)
#plot the bar graph
plt.bar(X_axis - 0.2, df_renew_r['1990'], 0.1, label = '1990')
plt.bar(X_axis - 0.1, df_renew_r['2000'], 0.1, label = '2000')
plt.bar(X_axis + 0, df_renew_r['2005'], 0.1, label = '2005')
plt.bar(X_axis + 0.1, df_renew_r['2010'], 0.1, label = '2010')
plt.bar(X_axis + 0.2, df_renew_r['2015'], 0.1, label = '2015')

## set the X&Y axis, lables, and legend
plt.title('Renewable Energy consumption')
plt.xlabel('Countries')
plt.ylabel('Percentages(%)')
plt.xticks(X_axis, Cnames)
plt.ylim(0,70) # set the limit for Y-axis
plt.legend()

plt.tight_layout() # for the layout
plt.savefig("Renewable Energy.jpg") 
plt.show()

print('-----------------------------------')


# BAR GRAPH    


print('-------------------------------')

#Download data from the World bank data and store the file where python file existing
df_energy = pd.read_csv("energy used.csv")
#print(df_energy)

#drop the columns names from the files
df_energy = df_energy.drop(['Country Code', 'Indicator Name', 'Indicator Code'],axis=1)
#print(df_energy)

#replace the '..' into None Values
df_energy = df_energy.replace('..', np.NaN)
#print(df_energy)

# Create a header
'''
header = df_energy.iloc[0].values.tolist()
df_energy.columns = header
print(df_energy)
'''
#acess perticular raw through index location function
df_energy = df_energy.iloc[[29,35,40,55,70,81,109,116,259], :] 
print(df_energy)

# Calling Counrty name column
Cnames = df_energy['Country Name']
print(Cnames)

# set the values for x-axis 
X_axis = np.arange(len(Cnames))

# set the figure size for better visualization
plt.figure(figsize=(10,8), dpi=500)
#plot the bar graph
plt.bar(X_axis - 0.2, df_energy['1990'], 0.1, label = '1990')
plt.bar(X_axis - 0.1, df_energy['2000'], 0.1, label = '2000')
plt.bar(X_axis + 0, df_energy['2005'], 0.1, label = '2005')
plt.bar(X_axis + 0.1, df_energy['2010'], 0.1, label = '2010')
plt.bar(X_axis + 0.2, df_energy['2015'], 0.1, label = '2015')

## set the X&Y axis, lables, and legend
plt.title('Energy Used')
plt.xlabel('Countries')
plt.ylabel('Energy Used(kg of oil equivalent per capita)')
plt.xticks(X_axis, Cnames)

plt.legend(loc = 'upper right')
plt.tight_layout()
plt.savefig("Energy Used.jpg")
plt.show()

print("For Line Graph -----------------------------------")

# for line graph

print("--------------------------")

#Download data from the World bank data and store the file where python file existing
df_nuclear = pd.read_csv("Electricity Produce from nuclear source.csv")
#print(df_nuclear)

#drop the columns names from the files
df_nuclear = df_nuclear.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)
print(df_nuclear)

#replace the '..' into None Values
df_nuclear = df_nuclear.replace('..', np.NaN) 

#Transpose through we can transpose columns names into raw
df_nuclear_t = pd.DataFrame.transpose(df_nuclear)
#print(df_renew_t)

#Create A Header
header = df_nuclear_t.iloc[0].values.tolist()
df_nuclear_t.columns = header
print(df_nuclear_t)

df_nuclear_t = df_nuclear_t.iloc[1:]
print(df_nuclear_t)

print(len(df_nuclear_t))
#selecting perticular countries
df_nuclear_t = df_nuclear_t[df_nuclear_t["Germany"].notna()]
df_nuclear_t = df_nuclear_t[df_nuclear_t["Italy"].notna()]
df_nuclear_t = df_nuclear_t[df_nuclear_t["India"].notna()]
df_nuclear_t = df_nuclear_t[df_nuclear_t["Japan"].notna()]
df_nuclear_t = df_nuclear_t[df_nuclear_t["United Kingdom"].notna()]
df_nuclear_t = df_nuclear_t[df_nuclear_t["Spain"].notna()]
df_nuclear_t = df_nuclear_t[df_nuclear_t["France"].notna()]
df_nuclear_t = df_nuclear_t[df_nuclear_t["World"].notna()]


print(len(df_nuclear_t))

# to get types of all columns
print(df_nuclear_t.dtypes)

# to get types of individual column

print(df_nuclear_t["Germany"])
print(df_nuclear_t["Italy"])
df_nuclear_t["Germany"] = pd.to_numeric(df_nuclear_t["Germany"])
df_nuclear_t["Italy"] = pd.to_numeric(df_nuclear_t["Italy"])
df_nuclear_t["India"] = pd.to_numeric(df_nuclear_t["India"])
df_nuclear_t["Japan"] = pd.to_numeric(df_nuclear_t["Japan"])
df_nuclear_t["United Kingdom"] = pd.to_numeric(df_nuclear_t["United Kingdom"])
df_nuclear_t["Spain"] = pd.to_numeric(df_nuclear_t["Spain"])
df_nuclear_t["France"] = pd.to_numeric(df_nuclear_t["France"])
df_nuclear_t["World"] = pd.to_numeric(df_nuclear_t["World"])
print(df_nuclear_t["Germany"])
print(df_nuclear_t["Italy"])


# convert index to int
df_nuclear_t.index = pd.to_numeric(df_nuclear_t.index)

#Set A figure size for better Visualization
plt.figure(figsize=(8,7))

#plot the line graph 
plt.plot(df_nuclear_t.index, df_nuclear_t["Germany"], label="Germany", ls ='dashed') # ls for linestyle here used dashed line
plt.plot(df_nuclear_t.index, df_nuclear_t["Italy"], label="Italy", ls ='dashed')
plt.plot(df_nuclear_t.index, df_nuclear_t["India"], label="India", ls ='dashed')
plt.plot(df_nuclear_t.index, df_nuclear_t["Japan"], label="Japan", ls ='dashed')
plt.plot(df_nuclear_t.index, df_nuclear_t["United Kingdom"], label="United Kingdom",ls='dashed')
plt.plot(df_nuclear_t.index, df_nuclear_t["Spain"], label="Spain",ls='dashed')
plt.plot(df_nuclear_t.index, df_nuclear_t["France"], label="France",ls='dashed')
plt.plot(df_nuclear_t.index, df_nuclear_t["World"], label="World",ls='dashed')

# set the X&Y axis, lables, and legend
plt.title("Electricity Production from Nuclear Sources")
plt.xlim(1975,2015)
plt.xlabel("year")
plt.ylabel("percentage(%)")

plt.legend(loc='upper right')
plt.savefig("nuclear.jpg") #save the figure
plt.show()


print("for second line graph -----------------------------------")

# for the line Graph

print("-------------------------------------------")


#Download data from the World bank data and store the file where python file existing
df_renew = pd.read_csv("Electricity Produce From Oil Source.csv")
#print(df_renew)

#drop the columns names from the files
df_renew = df_renew.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)
#print(df_renew)

#replace the '..' into None Values
df_renew = df_renew.replace('..', np.NaN) 

#Transpose through we can transpose columns names into raw
df_renew_t = pd.DataFrame.transpose(df_renew)
#print(df_renew_t)

#Create A Header
header = df_renew_t.iloc[0].values.tolist()
df_renew_t.columns = header
print(df_renew_t)


df_renew_t = df_renew_t.iloc[1:]
print(df_renew_t)

print(len(df_renew_t))
df_renew_t = df_renew_t[df_renew_t["Germany"].notna()]
df_renew_t = df_renew_t[df_renew_t["Italy"].notna()]
df_renew_t = df_renew_t[df_renew_t["India"].notna()]
df_renew_t = df_renew_t[df_renew_t["Japan"].notna()]
df_renew_t = df_renew_t[df_renew_t["United Kingdom"].notna()]
df_renew_t = df_renew_t[df_renew_t["Spain"].notna()]
df_renew_t = df_renew_t[df_renew_t["France"].notna()]
df_renew_t = df_renew_t[df_renew_t["World"].notna()]


print(len(df_renew_t))

# to get types of all columns
print(df_renew_t.dtypes)

# to get types of individual column

print(df_renew_t["Germany"])
print(df_renew_t["Italy"])
df_renew_t["Germany"] = pd.to_numeric(df_renew_t["Germany"])
df_renew_t["Italy"] = pd.to_numeric(df_renew_t["Italy"])
df_renew_t["India"] = pd.to_numeric(df_renew_t["India"])
df_renew_t["Japan"] = pd.to_numeric(df_renew_t["Japan"])
df_renew_t["United Kingdom"] = pd.to_numeric(df_renew_t["United Kingdom"])
df_renew_t["Spain"] = pd.to_numeric(df_renew_t["Spain"])
df_renew_t["France"] = pd.to_numeric(df_renew_t["France"])
df_renew_t["World"] = pd.to_numeric(df_renew_t["World"])

#print(df_renew_t["Germany"])
#print(df_renew_t["Italy"])

# convert index to int
df_renew_t.index = pd.to_numeric(df_renew_t.index)

#for line Ploting
plt.figure(figsize=(8,7)) # set figure size

# plot the line graph using pyplot
plt.plot(df_renew_t.index, df_renew_t["Germany"], label="Brazil", ls ='dashed')  # ls for line style for Dashed line
plt.plot(df_renew_t.index, df_renew_t["Italy"], label="China", ls ='dashed')
plt.plot(df_renew_t.index, df_renew_t["India"], label="India", ls ='dashed')
plt.plot(df_renew_t.index, df_renew_t["Japan"], label="Japan", ls ='dashed')
plt.plot(df_renew_t.index, df_renew_t["United Kingdom"], label="United Kingdom",ls='dashed')
plt.plot(df_renew_t.index, df_renew_t["Spain"], label="Spain",ls='dashed')
plt.plot(df_renew_t.index, df_renew_t["France"], label="France",ls='dashed')
plt.plot(df_renew_t.index, df_renew_t["World"], label="World",ls='dashed')

#print title for the Line Graph
plt.title('Electricity Production from Oil Sources')

# Set the X&Y Lables , axis and legend
plt.xlim(1980,2015)
plt.xlabel("year")
plt.ylabel("Percentage(%)")

plt.legend(loc='upper right')
plt.savefig('Oil.jpg')
plt.show()


print("for Table-------------------------------")

# for Table 

print('-----------------------------------------')

df_population = pd.read_csv('Population.csv')
print(df_population)

#drop the columns names from the files
df_population = df_population.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)
#print(df_renew)

#replace the '..' into None Values
df_population = df_population.replace('..', np.NaN) 
'''
#Transpose through we can transpose columns names into raw
df_population_t = pd.DataFrame.transpose(df_population)
#print(df_renew_t)

#Create A Header
header = df_population_t.iloc[0].values.tolist()
df_population_t.columns = header
print(df_population_t)

df_population_t = df_population_t.iloc[1:]
print(df_population_t)
'''
df_population = df_population.iloc[[29,35,40,55,70,81,109,116,259], :]
print(df_population)