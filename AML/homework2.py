import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#In the NP library, randn is normally distributed, in the doc string you can see it uses "standard normal" distrobution
#rand is uniformly distributed as seen in the doc string
arr_five_hundred = np.random.randn(500)
arr_thousand = np.random.randn(1000)
arr_five_thousand = np.random.randn(5000)
arr_uni_five_hundred =np.random.rand(500)
arr_uni_thousand = np.random.rand(1000)
arr_uni_five_thousand = np.random.rand(5000)

#Make a simple function so I dont have to type out this block every time
def plot_histogram(data, title, xlabel, ylabel, bins=30):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

#Put the data in a dictionary to loop through
data_dict = {
    "Normal Distribution 500": arr_five_hundred,
    "Normal Distribution 1000": arr_thousand,
    "Normal Distribution 5000": arr_five_thousand,
    "Uniform Distribution 500": arr_uni_five_hundred,
    "Uniform Distribution 1000": arr_uni_thousand,
    "Uniform Distribution 5000": arr_uni_five_thousand
}
#Easy to loop through all 6 arrays this way
for title, data in data_dict.items():
    plot_histogram(data, title, 'Value', 'Frequency')
    
    
#For question 2

from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Dowload and prepare the data

#This sets the https path to the git repository and the read_csv path is actually able to acess that 
#repository and download the data if given the correct path
data_root= 'https://github.com/ageron/data/raw/main/'
lifesat = pd.read_csv(data_root + 'lifesat/lifesat.csv')
#Split data into our x and ys, pull just the values, and use .values to get series.
X = lifesat[['GDP per capita (USD)']].values
y = lifesat['Life satisfaction'].values

#Make a plot using Matplotlib
lifesat.plot(kind='scatter', grid=True, x='GDP per capita (USD)', y='Life satisfaction')
#Set specific values on the axis
plt.axis([23_500,62_500,4,9])
#To print the plot to the screen
plt.show()

#Create an instance of the linear regression class from sklearn
model = LinearRegression()
#Fit the model to the data
model.fit(X, y)

#Creating new prediction
X_new = [[37_655.2]] #Cyprus GDP per capita
print(model.predict(X_new)) 

#Using KNN regressing, predicting the value based on the 3 nearest data points to the point we are predicting
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)
print(model.predict(X_new))