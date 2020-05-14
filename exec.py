#%% OPEN USING VISUAL STUDIO CODE
# Splitting source xlsx file to smaller chunks for as GitHub repo 
# has file size limit.
import pandas as pd
import numpy as np
# !pip install xlrd 
NUM_CLUSTERS = 10
# %%
df = pd.read_excel("CORE_DATA.xlsx")
dfs = np.array_split(df, 5)
for idx in range(len(dfs)):
    dfs[idx].to_pickle("core_data_chunk_"+str(idx)+".pkl")

# %%
# load data from chunks
pickles = []
for dirname, _, filenames in os.walk('./'):
    for filename in filenames:
        if (filename[-3:] == 'pkl'):
            pickles.append(os.path.join(dirname, filename))

df_chunks = []
for pkl in pickles:
    _df = pd.read_pickle(pkl)
    df_chunks.append(_df)

df = pd.concat(df_chunks)
genes = list(df.iloc[:,0])              # set aside gene names
df = df.drop(df.columns[0], axis =1)    # remove all non-numeric data
# %%
# import dependencies
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler

# %% Data preprocessing
mat = df.values
scaler = MinMaxScaler()
scaler.fit(mat)
mat_sc = scaler.transform(mat)

# %% Basic K means
kmeans = KMeans(n_clusters=NUM_CLUSTERS)
kmeans.fit(mat)
labels = kmeans.labels_
results_km = pd.DataFrame([df.index,labels]).T

#%% Hierarchical
hierarchical = AgglomerativeClustering(n_clusters=NUM_CLUSTERS, affinity='euclidean', compute_full_tree='auto', linkage='ward', distance_threshold=None).fit(mat)
hierarchical
labels = hierarchical.labels_
results_hc = pd.DataFrame([df.index,labels]).T

# %%
df['gene_name'] = pd.Series(genes, index=df.index)
df['k_means_cluster_id'] = results_km.loc[:,1]
df['hierarchical_cluster_id'] = results_hc.loc[:,1]
print(df)
df.to_csv('results.csv')