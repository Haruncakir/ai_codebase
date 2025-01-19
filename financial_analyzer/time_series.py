import pandas as pd
import datasets

# Load TSLA dataset
tesla_data = datasets.load_dataset('tsla-historic-prices')
tesla_df = pd.DataFrame(tesla_data['train'])

# Display the first few rows
print(tesla_df.head())

# Convert the Date column to datetime type
tesla_df['Date'] = pd.to_datetime(tesla_df['Date'])

# Display the first few rows to verify the change
print(tesla_df.head())

'''
        Date      Open      High       Low     Close  Adj Close     Volume
0 2010-06-29  1.266667  1.666667  1.169333  1.592667   1.592667  281494500
1 2010-06-30  1.719333  2.028000  1.553333  1.588667   1.588667  257806500
2 2010-07-01  1.666667  1.728000  1.351333  1.464000   1.464000  123282000
3 2010-07-02  1.533333  1.540000  1.247333  1.280000   1.280000   77097000
4 2010-07-06  1.333333  1.333333  1.055333  1.074000   1.074000  103003500
'''

# Setting the Date column as the index is crucial for time series operations.
# It allows us to sort the data chronologically and makes it easier to slice by specific dates.

# Set Date column as the index
tesla_df.set_index('Date', inplace=True)

# Display the first few rows to verify the change
print(tesla_df.head())

'''
                Open      High       Low     Close  Adj Close     Volume
Date                                                                    
2010-06-29  1.266667  1.666667  1.169333  1.592667   1.592667  281494500
2010-06-30  1.719333  2.028000  1.553333  1.588667   1.588667  257806500
2010-07-01  1.666667  1.728000  1.351333  1.464000   1.464000  123282000
2010-07-02  1.533333  1.540000  1.247333  1.280000   1.280000   77097000
2010-07-06  1.333333  1.333333  1.055333  1.074000   1.074000  103003500
'''

# Sort the DataFrame based on the index in descending order
tesla_df.sort_index(ascending=False, inplace=True)

# Display the first few rows to verify the change
print(tesla_df.head())

'''
                  Open        High  ...   Adj Close     Volume
Date                                ...                       
2023-10-13  258.899994  259.600006  ...  251.119995  102073800
2023-10-12  262.920013  265.410004  ...  258.869995  111508100
2023-10-11  266.200012  268.600006  ...  262.989990  103706300
2023-10-10  257.750000  268.940002  ...  263.619995  122656000
2023-10-09  255.309998  261.359985  ...  259.670013  101377900
'''


