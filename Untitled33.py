#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd


# In[19]:


df= pd.read_csv("D:\internship\mcdonalds.csv")


# In[20]:


print(df)


# In[21]:


df.head(0)


# In[22]:


df.shape


# In[23]:


df.head(3)


# In[24]:


df = pd.DataFrame(df)
df=df.iloc[:, :11]


# In[25]:


df = (df == 'Yes').astype(int)
average_values = df.mean().round(2)
print(average_values)        


# In[26]:


import numpy as np
matrix = np.cov(df, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(matrix)
sorted_index = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_index]
eigenvectors = eigenvectors[:, sorted_index]
explained_variance = eigenvalues / np.sum(eigenvalues)
cumulative_explained_variance = np.cumsum(explained_variance)
std_devs = np.sqrt(eigenvalues)
pca_summary = pd.DataFrame({
    'Standard Deviation': std_devs,
    'Proportion of Variance': explained_variance,
    'Cumulative Proportion': cumulative_explained_variance
})

print("PCA Summary:")
print(pca_summary.round(4))


# In[27]:


std_devs_rounded = np.round(std_devs, 1)
print(std_devs_rounded)


# In[28]:


rotation_matrix = pd.DataFrame(eigenvectors,index=df.columns)* -1
print(rotation_matrix.round(3))


# In[29]:


import numpy as np
import matplotlib.pyplot as plt
consumers_projected = np.dot(df, eigenvectors[:, :2])
plt.scatter(consumers_projected[:, 0], consumers_projected[:, 1], color='grey')
for  i,(comp,var) in enumerate(zip(eigenvectors[:, :2], df.columns)):
    plt.arrow(0, 0, -comp[0], -comp[1], color='r', alpha=0.9, width=0.01)
    plt.text(-comp[0] * 1.15, -comp[1] * 1.15, var, color='r', ha='center', va='center', fontsize=10)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Projection of Consumers and Segmentation Variables')
plt.grid(True)
plt.legend(loc='upper left')
plt.show()


# In[30]:


from sklearn.cluster import KMeans
np.random.seed(1234)
k_values = range(2, 9)
within_sum_of_squares = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(df)
    within_sum_of_squares.append(kmeans.inertia_)


plt.bar(k_values, within_sum_of_squares)
plt.xlabel('Number of Segments')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('Scree Plot for Determining Number of Segments')
plt.xticks(k_values)
plt.show()

rate_of_change = np.diff(within_sum_of_squares)
elbow_index = np.argmax(rate_of_change) + 1  
optimal_k = k_values[elbow_index]

print("Optimal number of segments:", optimal_k)


# In[31]:


from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample
x = df.to_numpy() 
np.random.seed(1234)
def bootstrap_kmeans(X, k, n_repeats, n_bootstraps):
   
    kmeans = KMeans(n_clusters=k, n_init=n_repeats, random_state=1234)
    original_labels = kmeans.fit_predict(X)
    stability_scores = []

    for _ in range(n_bootstraps):
        bootstrap_sample = resample(X, n_samples=len(X), random_state=1234)
        bootstrap_kmeans = KMeans(n_clusters=k, n_init=n_repeats, random_state=1234)
        bootstrap_labels = bootstrap_kmeans.fit_predict(bootstrap_sample)
        ari_score = adjusted_rand_score(original_labels, bootstrap_labels)
        stability_scores.append(ari_score)

    return stability_scores


k_values = range(2, 9)
global_stability = []

n_repeats = 10
n_bootstraps = 100

for k in k_values:
    stability_scores = bootstrap_kmeans(MD_x, k, n_repeats, n_bootstraps)
    global_stability.append(stability_scores)


plt.boxplot(global_stability, labels=k_values)
plt.xlabel('Number of Segments')
plt.ylabel('Adjusted Rand Index')
plt.ylim(0.4, 1.0)  
plt.title('Global Stability Analysis')
plt.show()


# In[32]:


df_binary = (df == 'Yes').astype(int).to_numpy()
np.random.seed(1234)
noise = np.random.normal(0, 0.1, df_binary.shape)
df_noisy = df_binary + noise
df_noisy = np.clip(df_noisy, 0, 1)
kmeans = KMeans(n_clusters=4, n_init=10, random_state=1234)
kmeans.fit(df_noisy)
cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_


distances = np.zeros(len(df_noisy))
for i, point in enumerate(df_noisy):
    cluster_center = cluster_centers[cluster_labels[i]]
    distances[i] = np.linalg.norm(point - cluster_center)


plt.figure(figsize=(10, 6))
for cluster_id in range(4):
    plt.hist(distances[cluster_labels == cluster_id], bins=np.linspace(0, 1, 50), alpha=0.6, label=f'Cluster {cluster_id + 1}')

plt.xlabel('Distance to Cluster Center')
plt.ylabel('Frequency')
plt.title('Gorge Plot for 4-Segment Solution')
plt.legend()
plt.xlim(0, 1)
plt.show()


# In[ ]:




