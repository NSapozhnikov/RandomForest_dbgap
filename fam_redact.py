import pandas as pd
from collections import Counter

pd_df = pd.read_csv('orig_data.clean.pruned.fam', 
                     sep=' ', 
                     lineterminator='\n', 
                     header=None)
pd_df.rename(columns={pd_df.columns[0]:'IID', pd_df.columns[5]:'pheno'}, 
                      inplace = True)
print(set(pd_df['pheno']))

targets_df = pd.read_csv('negativeSelected_NearMiss3_3.csv',
                          lineterminator='\n')
targets_df['pheno'] = 3
# print(targets_df)

pd_df = pd_df.merge(targets_df, on='IID', how='left')

# pd_df.loc[pd_df['pheno_y'] == 3, pd_df['pheno_x']] = 3
pd_df['pheno_x'].mask(pd_df['pheno_y'] == 3, 3, inplace=True)
print(pd_df)

pd_df.drop('pheno_y', axis=1,inplace=True)
print(set(pd_df['pheno_x']))
print(Counter(pd_df['pheno_x']))

pd_df.to_csv('data.clean.pruned.fam',
              sep=' ',
              header=False,
              index=False)
