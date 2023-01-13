import pandas as pd
import numpy as np
from collections import Counter
import imblearn as imb
from sklearn.preprocessing import OneHotEncoder

### Reading .pkl file
num_chr = '1'
pd_df = pd.read_pickle(f"/mnt/wd/nsap/imp2/chr{num_chr}.pkl")
print('Original DF: ', pd_df, sep='\n')

## Saving 1st two columns
classes = pd_df['PHENOTYPE']
pd_df_out = pd_df[['IID','PHENOTYPE']].copy()
pd_df.drop(['IID','PHENOTYPE'], axis=1, inplace=True)

### One-Hot encode
print('Performing one-hot encoding...')
enc = OneHotEncoder(sparse=False, handle_unknown='error')
enc.fit(pd_df)
pd_df_enc = enc.transform(pd_df)
pd_df_columns = enc.get_feature_names_out()
print('Encoded...')

### NearMiss-1 selects the positive samples for which the average distance to the N
### closest samples of the negative class is the smallest.
print('Performing NearMiss undersampling...')
undersample = imb.under_sampling.NearMiss(sampling_strategy='majority',
                                          version=1,
                                          n_neighbors=3, 
                                          n_jobs=12)
pd_df_enc, classes = undersample.fit_resample(pd_df_enc, classes)
counter = Counter(classes)
print('Classes counter: ', counter)
resampled_idx = undersample.sample_indices_
pd_df_res = pd.DataFrame(pd_df_enc, columns=pd_df_columns, index=resampled_idx)

# ### Oversampling technique SMOTEN (Synthetic Minority Over-sampling Technique for Nominal)
# print(f"Original class counts: {Counter(classes)}")
# smoten_over = imb.over_sampling.SMOTEN(n_jobs=6)
# pd_df_res, classes_res = smoten_over.fit_resample(pd_df_enc, classes)
# print(f"Class counts after resampling {Counter(classes_res)}")
# print('pd_df_res: ', pd_df_res, sep='\n')

# ### Save new DF
# pd_df_out = pd.merge(pd_df_out, pd_df_res, left_index=True, right_index=True)
# print(pd_df_out)
# pd_df_out.to_pickle(f"/mnt/wd/nsap/imp2/chr{num_chr}_nearmiss.pkl")
