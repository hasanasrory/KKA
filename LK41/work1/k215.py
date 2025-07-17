import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Muat data
df = pd.read_csv('Data_Penjualan_Sepeda_Motor_2023-2025.csv')  # Ganti sesuai lokasi file

# Agregasi data per bulan
pivot_df = df.pivot_table(index='Bulan', columns='Produsen', values='Penjualan_Unit', aggfunc='sum').reset_index()
pivot_df.fillna(0, inplace=True)

# Fitur yang digunakan
features = pivot_df[['Honda', 'Yamaha', 'Suzuki', 'Kawasaki', 'TVS', 'Lainnya']]

# Normalisasi
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Klasterisasi
kmeans = KMeans(n_clusters=3, random_state=42)
pivot_df['Cluster'] = kmeans.fit_predict(scaled_features)

# Tampilkan hasil
print(pivot_df.head())

# Visualisasi (2D dengan PCA jika diperlukan)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
components = pca.fit_transform(scaled_features)

plt.figure(figsize=(8,6))
plt.scatter(components[:, 0], components[:, 1], c=pivot_df['Cluster'], cmap='tab10')
plt.title('Klasterisasi Bulan Berdasarkan Penjualan Sepeda Motor')
plt.xlabel('PCA Komponen 1')
plt.ylabel('PCA Komponen 2')
plt.grid(True)
plt.show()

