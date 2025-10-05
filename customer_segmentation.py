import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = pd.read_excel(r"D:\DATA ANALYSIS\online_retail_II.xlsx")
print("Dataset Loaded Successfully!\n")
print(data.head())
print("\nActual Column Names:")
for col in data.columns:
    print(f"'{col}'")
data.columns = data.columns.str.strip().str.replace(' ', '').str.lower()
print("\nCleaned Column Names:")
print(data.columns.tolist())
if 'customerid' not in data.columns:
    raise KeyError("'CustomerID' column not found even after cleaning. Please verify the Excel file.")
data = data.dropna(subset=['customerid'])
price_column = None
if 'unitprice' in data.columns:
    price_column = 'unitprice'
elif 'price' in data.columns:
    price_column = 'price'
else:
    raise KeyError("Neither 'UnitPrice' nor 'Price' column found!")
data = data[(data['quantity'] > 0) & (data[price_column] > 0)]
data['totalprice'] = data['quantity'] * data[price_column]
rfm = data.groupby('customerid').agg({
    'invoicedate': lambda x: (data['invoicedate'].max() - x.max()).days,
    'invoice': 'count',
    'totalprice': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
print("\nRFM Table Created:")
print(rfm.head())
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal k')
plt.show()
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("\nCluster Summary:")
print(cluster_summary)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm, palette='tab10', s=80)
plt.title('Customer Segmentation (K-Means Clustering)')
plt.show()
