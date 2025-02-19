import pandas as pd
import datasets

# Load the dataset
data = datasets.load_dataset('tsla-historic-prices')
tesla_df = pd.DataFrame(data['train'])
print(tesla_df.head())

'''
         Date      Open      High       Low     Close  Adj Close     Volume
0  2010-06-29  1.266667  1.666667  1.169333  1.592667   1.592667  281494500
1  2010-06-30  1.719333  2.028000  1.553333  1.588667   1.588667  257806500
2  2010-07-01  1.666667  1.728000  1.351333  1.464000   1.464000  123282000
3  2010-07-02  1.533333  1.540000  1.247333  1.280000   1.280000   77097000
4  2010-07-06  1.333333  1.333333  1.055333  1.074000   1.074000  103003500
'''

# Creating the High-Low feature
tesla_df['High-Low'] = tesla_df['High'] - tesla_df['Low']

# Creating the Price-Open feature
tesla_df['Price-Open'] = tesla_df['Close'] - tesla_df['Open']

# Displaying the new features
print(tesla_df[['High-Low', 'Price-Open']].head())

'''
   High-Low  Price-Open
0  0.497334    0.326000
1  0.474667   -0.130666
2  0.376667   -0.202667
3  0.292667   -0.253333
4  0.278000   -0.259333
'''


