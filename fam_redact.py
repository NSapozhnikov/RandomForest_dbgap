import pandas as pd

pd_df = pd.read_csv('orig_data.clean.pruned.fam', 
                     sep=' ', 
                     lineterminator='\n', 
                     header=None)
pd_df.rename(columns={pd_df.columns[0]:'IID', pd_df.columns[5]:'pheno'}, 
                      inplace = True)
print(pd_df)

targets_df = pd.read_csv('positiveSelected_NearMiss1.csv',
                          lineterminator='\n')
targets_df['pheno'] = 3
print(targets_df)

pd_df = pd_df.merge(targets_df, on='IID', how='left')
print(pd_df)

pd_df['pheno_x'].update(pd_df['pheno_y'])
pd_df.drop('pheno_y', axis=1,inplace=True)
print(pd_df)
pd_df.to_csv('data.clean.pruned.fam',
              sep=' ',
              header=False,
              index=False)
              
#kogjslgjskgd
