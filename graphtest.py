# Python program to illustrate
# Plotting categorical scatter 
# plots with Seaborn
 
# importing the required module
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd 
 
#x axis values
x =['sun', 'mon', 'fri', 'sat', 'tue', 'wed', 'thu']
 
# y axis values
y =[5, 6.7, 4, 6, 2, 4.9, 1.8]
 
# plotting strip plot with seaborn
ax = sns.stripplot(x, y);
 
# giving labels to x-axis and y-axis
ax.set(xlabel ='Days', ylabel ='Amount_spend')
 
# giving title to the plot
plt.title('My first graph');
# cm = np.matrix([[1,2,3],
# 		[3,2,1],
# 		[2,1,3]])
# df = pd.DataFrame(cm)
# plt.figure(figsize=(9,9))
# sns.heatmap(cm, annot=True, cmap="Blues_r")
print("PLOTING")
# function to show plot
plt.show()
print("Finish Ploting")