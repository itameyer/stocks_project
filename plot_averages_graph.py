import pandas as pd
import matplotlib.pyplot as plt

# Read CSV and plot column as dots
df = pd.read_csv('snp500_companies/averages.csv')
y = df['Average']
x = range(len(y))

plt.scatter(x, y)
plt.show()