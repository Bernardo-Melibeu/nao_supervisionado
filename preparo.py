# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# !pip freeze

# %%
df_original=pd.read_csv("data/Country-data.csv")
df_original.info
df=df_original.copy()
LISTA=["child_mort","exports","health","imports","income","inflation","life_expec","total_fer","gdpp"]
N_CLUSTER=3

# %% [markdown]
# ## Escalando os dados

# %%
scaler=StandardScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(df[LISTA]),
                           columns=df[LISTA].columns,
                           index=df[LISTA].index)

# %% [markdown]
# ## PCA

# %%
pca_todos=PCA(n_components=9)
pca_todos=pca_todos.fit(scaled_data)

# %%
pca=PCA(n_components=4)
pca_scaled=pca.fit_transform(scaled_data)

# %%
n_comp=pca_scaled.shape[1]


