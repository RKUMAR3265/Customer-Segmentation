
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
data = pd.read_excel(r"D:\DATA ANALYSIS\online_retail_II.xlsx")
print("✅ Dataset Loaded Successfully!\n")
print(data.head())

# Step 2: Print actual column names to check structure
print("\n🔹 Actual Column Names:")
for col in data.columns:
    print(f"'{col}'")

# Step 3: Clean column names (remove spaces and make lowercase for uniformity)
data.columns = data.columns.str.strip().str.replace(' ', '').str.lower()
print("\n✅ Cleaned Column Names:")
print(data.columns.tolist())

# Step 4: Check if CustomerID exists after cleaning
if 'customerid' not in data.columns:
    raise KeyError("❌ 'CustomerID' column not found even after cleaning. Please verify the Excel file.")

# Step 5: Data Cleaning
# Drop missing CustomerID rows
data = data.dropna(subset=['customerid'])

# Remove negative or zero Quantity and Price/UnitPrice
price_column = None
if 'unitprice' in data.columns:
    price_column = 'unitprice'
elif 'price' in data.columns:
    price_column = 'price'
else:
    raise KeyError("❌ Neither 'UnitPrice' nor 'Price' column found!")

data = data[(data['quantity'] > 0) & (data[price_column] > 0)]
data['totalprice'] = data['quantity'] * data[price_column]

# Step 6: Prepare RFM (Recency, Frequency, Monetary)
rfm = data.groupby('customerid').agg({
    'invoicedate': lambda x: (data['invoicedate'].max() - x.max()).days,
    'invoice': 'count',
    'totalprice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
print("\n✅ RFM Table Created:")
print(rfm.head())

# Step 7: Normalize RFM data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Step 8: Elbow Method to find optimal number of clusters
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

# Step 9: Apply KMeans Clustering (use k=4 for example)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Step 10: Cluster Analysis
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("\n📊 Cluster Summary:")
print(cluster_summary)

# Step 11: Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm, palette='tab10', s=80)
plt.title('Customer Segmentation (K-Means Clustering)')
plt.show()
# -------------------------------
# CUSTOMER SEGMENTATION USING K-MEANS (FINAL FIXED VERSION)
# -------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load dataset
data = pd.read_excel(r"D:\DATA ANALYSIS\online_retail_II.xlsx")
print("✅ Dataset Loaded Successfully!\n")
print(data.head())

# Step 2: Print actual column names to check structure
print("\n🔹 Actual Column Names:")
for col in data.columns:
    print(f"'{col}'")

# Step 3: Clean column names (remove spaces and make lowercase for uniformity)
data.columns = data.columns.str.strip().str.replace(' ', '').str.lower()
print("\n✅ Cleaned Column Names:")
print(data.columns.tolist())

# Step 4: Check if CustomerID exists after cleaning
if 'customerid' not in data.columns:
    raise KeyError("❌ 'CustomerID' column not found even after cleaning. Please verify the Excel file.")

# Step 5: Data Cleaning
# Drop missing CustomerID rows
data = data.dropna(subset=['customerid'])

# Remove negative or zero Quantity and Price/UnitPrice
price_column = None
if 'unitprice' in data.columns:
    price_column = 'unitprice'
elif 'price' in data.columns:
    price_column = 'price'
else:
    raise KeyError("❌ Neither 'UnitPrice' nor 'Price' column found!")

data = data[(data['quantity'] > 0) & (data[price_column] > 0)]
data['totalprice'] = data['quantity'] * data[price_column]

# Step 6: Prepare RFM (Recency, Frequency, Monetary)
rfm = data.groupby('customerid').agg({
    'invoicedate': lambda x: (data['invoicedate'].max() - x.max()).days,
    'invoice': 'count',
    'totalprice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
print("\n✅ RFM Table Created:")
print(rfm.head())

# Step 7: Normalize RFM data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Step 8: Elbow Method to find optimal number of clusters
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

# Step 9: Apply KMeans Clustering (use k=4 for example)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Step 10: Cluster Analysis
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print("\n📊 Cluster Summary:")
print(cluster_summary)

# Step 11: Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Recency', y='Monetary', hue='Cluster', data=rfm, palette='tab10', s=80)
plt.title('Customer Segmentation (K-Means Clustering)')
plt.show()
