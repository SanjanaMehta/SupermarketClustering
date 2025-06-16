#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans


# ### Load Data and Basic Exploratory Analysis

# In[2]:


df=pd.read_csv(r"C:\Users\Sanjana\Downloads\marketing_campaign.csv",sep="\t")
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# ### Data Cleaning and EDA

# In[5]:


df['Education'].unique()


# In[6]:


df['Marital_Status'].unique()


# In[7]:



un=df['Year_Birth'].unique()
sort_values=np.sort(un)
sort_values


# In[8]:


cols_drop=['AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Z_CostContact','Z_Revenue']
df=df.drop(columns=cols_drop,axis=1)


# In[9]:


df.describe()


# In[10]:


upper_lim=df.Income.mean()+3*df.Income.std()


# In[11]:


lower_lim=df.Income.mean()-3*df.Income.std()


# In[12]:


sns.boxplot(df['Income'])


# In[13]:


maxth=df['Income'].quantile(0.95)
minth=df['Income'].quantile(0.05)
df=df[((minth<df['Income'])&(df['Income']<maxth))]
df.describe()


# In[14]:


df['Dt_Customer']=pd.to_datetime(df['Dt_Customer'])


# In[15]:


df.info()


# ### Feature Engineering

# In[17]:


df['Age']=2025-df['Year_Birth']
col_to_drop=['Year_Birth']
df=df.drop(columns=col_to_drop,axis=1)


# In[18]:


df['Amount Purchased']=(df['MntWines']+df['MntFruits']+df['MntMeatProducts']+df['MntFishProducts']+df['MntSweetProducts']+df['MntGoldProds'])


# In[19]:


df['Children']=df['Kidhome']+df['Teenhome']


# In[20]:


df=df.drop('Teenhome',axis=1)
df=df.drop('Kidhome',axis=1)


# In[21]:


df


# In[22]:


df['Marital_Status'] = df['Marital_Status'].replace(['Absurd', 'YOLO', 'Alone'], 'Other')

# Print the updated value counts to verify
print(df['Marital_Status'].value_counts())


# In[23]:


corr=df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr,annot=True)
plt.show()


# ### Dimensionality Reduction

# In[24]:


s=(df.dtypes=='object')
objcols=list(s[s].index)


# In[25]:


LE=LabelEncoder()
for i in objcols:
    df[i]=df[[i]].apply(LE.fit_transform)


# In[26]:


df.info()


# In[27]:


t=(df.dtypes=='datetime64[ns]')
dtcols=list(t[t].index)
for i in dtcols:
    df[i]=df[[i]].apply(LE.fit_transform)


# In[28]:


ds=df.copy()
scaler = StandardScaler()
scaler.fit(ds)
scaled_ds = pd.DataFrame(scaler.transform(ds),columns= ds.columns )


# ### Principal Component Analysis

# In[29]:


pca=PCA(n_components=3)
pca.fit(scaled_ds)
PCA_ds=pd.DataFrame(pca.transform(scaled_ds), columns=(["col1","col2", "col3"]))
PCA_ds.describe().T


# ### KMeans Clustering

# In[31]:




elbow=KElbowVisualizer(KMeans(),K=10)
elbow.fit(PCA_ds)


# In[33]:


kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(PCA_ds)

# Correct way to get cluster labels
labels = kmeans.labels_

# Add labels to DataFrame
PCA_ds["Clusters"] = labels


# In[34]:




# Scatter plot of the clusters
plt.figure(figsize=(8, 5))
plt.scatter(PCA_ds.iloc[:, 0], PCA_ds.iloc[:, 1], 
            c=PCA_ds["Clusters"], cmap='viridis', s=50)

plt.title('KMeans Clustering (PCA Reduced Data)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()


# ### Evaluating Model

# In[38]:


df['Clusters'] = PCA_ds['Clusters'].values


# In[39]:


pal=['#480b59','#517593','#f6e639','#47b97e']
sns.countplot(df["Clusters"],palette= pal)


# In[40]:


sns.scatterplot(data=df,x=df['Amount Purchased'],y=df['Income'],hue=df['Clusters'],palette=pal)


# In[41]:


df.to_csv("Supermarket Data.csv",index=False)

