import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy import stats

# Cấu hình hiển thị
plt.rcParams['figure.figsize'] = (15, 10)
plt.style.use('seaborn-v0_8')
#Bai 1
#1
df_sports = pd.read_csv(r'C:\LuyenTap\Labs\Lab3\ITA105_Lab_3_Sports.csv')

print("--- 1. Kiểm tra dữ liệu ---")
print(f"Shape: {df_sports.shape}")
print("\nMissing values:")
print(df_sports.isnull().sum())
print("\nThống kê mô tả:")
print(df_sports.describe())

#2
cols_to_plot = ['chieu_cao_cm', 'can_nang_kg', 'toc_do_100m_s', 'so_phut_thi_dau']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
for i, col in enumerate(cols_to_plot):
    sns.histplot(df_sports[col], kde=True, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'Phân phối {col}')

plt.tight_layout()
plt.show()

#4
min_max_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(
    min_max_scaler.fit_transform(df_sports[cols_to_plot]), 
    columns=[f'{c}_minmax' for c in cols_to_plot]
)

#4
std_scaler = StandardScaler()
df_zscore = pd.DataFrame(
    std_scaler.fit_transform(df_sports[cols_to_plot]), 
    columns=[f'{c}_zscore' for c in cols_to_plot]
)

#5
col_test = 'chieu_cao_cm'

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gốc
sns.histplot(df_sports[col_test], kde=True, color='blue', ax=axes[0])
axes[0].set_title(f'Original {col_test}')

# Min-Max
sns.histplot(df_minmax[f'{col_test}_minmax'], kde=True, color='green', ax=axes[1])
axes[1].set_title(f'Min-Max Scaled (Range 0-1)')

# Z-Score
sns.histplot(df_zscore[f'{col_test}_zscore'], kde=True, color='red', ax=axes[2])
axes[2].set_title(f'Z-Score Scaled (Mean=0, Std=1)')

plt.show()

print("\n--- Kiểm tra sau chuẩn hóa ---")
print(f"Z-Score Mean: {df_zscore[f'{col_test}_zscore'].mean():.2f}")
print(f"Z-Score Std: {df_zscore[f'{col_test}_zscore'].std():.2f}")

#Bai 2
plt.rcParams['figure.figsize'] = (15, 10)
plt.style.use('ggplot')

#1
df_health = pd.read_csv(r'C:\LuyenTap\Labs\Lab3\ITA105_Lab_3_Health.csv')

print(df_health.describe())

cols = ['BMI', 'huyet_ap_mmHg', 'nhip_tim_bpm', 'cholesterol_mg_dl']
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, col in enumerate(cols):
    sns.histplot(df_health[col], kde=True, ax=axes[0, i], color='skyblue')
    axes[0, i].set_title(f'Histogram {col}')
    
    sns.boxplot(y=df_health[col], ax=axes[1, i], color='lightcoral')
    axes[1, i].set_title(f'Boxplot {col}')

plt.tight_layout()
plt.show()
#2
print("\n--- 2. Phát hiện ngoại lệ (|Z| > 3) ---")
z_scores = np.abs(stats.zscore(df_health[cols]))
outliers = df_health[(z_scores > 3).any(axis=1)]
print(f"Số lượng bệnh nhân có chỉ số cực đoan: {len(outliers)}")
print(outliers)
#3
# Min-Max
mm_scaler = MinMaxScaler()
df_mm = pd.DataFrame(mm_scaler.fit_transform(df_health[cols]), columns=cols)

# Z-Score
std_scaler = StandardScaler()
df_std = pd.DataFrame(std_scaler.fit_transform(df_health[cols]), columns=cols)

#4
target_col = 'BMI'
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df_health[target_col], kde=True, ax=axes[0], color='blue').set_title("Original BMI")
sns.histplot(df_mm[target_col], kde=True, ax=axes[1], color='green').set_title("Min-Max Scaled")
sns.histplot(df_std[target_col], kde=True, ax=axes[2], color='red').set_title("Z-Score Scaled")

plt.show()

#Bai 3

plt.rcParams['figure.figsize'] = (15, 10)
plt.style.use('fivethirtyeight')

#1
df_fin = pd.read_csv('ITA105_Lab_3_Finance.csv')

print("--- 1. Kiểm tra ngoại lệ (Công ty cực lớn) ---")
print(df_fin[['doanh_thu_musd', 'loi_nhuan_musd']].describe())

plt.figure(figsize=(10, 6))
sns.boxplot(data=df_fin[['doanh_thu_musd', 'loi_nhuan_musd']])
plt.title("Boxplot Doanh thu và Lợi nhuận (Trước chuẩn hóa)")
plt.show()

#2
cols = ['doanh_thu_musd', 'loi_nhuan_musd']

# Min-Max
mm_scaler = MinMaxScaler()
df_mm = pd.DataFrame(mm_scaler.fit_transform(df_fin[cols]), columns=cols)

# Z-Score
std_scaler = StandardScaler()
df_std = pd.DataFrame(std_scaler.fit_transform(df_fin[cols]), columns=cols)

#3
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.scatterplot(data=df_fin, x='doanh_thu_musd', y='loi_nhuan_musd', ax=axes[0], color='blue')
axes[0].set_title("Gốc (Original Scale)")

sns.scatterplot(data=df_mm, x='doanh_thu_musd', y='loi_nhuan_musd', ax=axes[1], color='green')
axes[1].set_title("Sau Min-Max [0, 1]")

sns.scatterplot(data=df_std, x='doanh_thu_musd', y='loi_nhuan_musd', ax=axes[2], color='red')
axes[2].set_title("Sau Z-Score (Mean=0, Std=1)")

plt.tight_layout()
plt.show()

#Bai 4
plt.style.use('bmh')
plt.rcParams['figure.figsize'] = (15, 12)

#1
df_game = pd.read_csv(r'C:\LuyenTap\Labs\Lab3\ITA105_Lab_3_Gaming.csv')

print("--- 1. Kiểm tra Missing Values ---")
print(df_game.isnull().sum())

print("\n--- Thống kê mô tả (Chú ý Max vs Mean) ---")
print(df_game.describe())

cols_gaming = ['gio_choi', 'diem_tich_luy', 'so_level', 'so_vat_pham']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for i, col in enumerate(cols_gaming):
    sns.histplot(df_game[col], kde=True, ax=axes[i//2, i%2], color='purple')
    axes[i//2, i%2].set_title(f'Phân phối {col} gốc')

plt.tight_layout()
plt.show()

#2
mm_scaler = MinMaxScaler()
df_game_mm = pd.DataFrame(mm_scaler.fit_transform(df_game[cols_gaming]), columns=cols_gaming)

# Z-Score
std_scaler = StandardScaler()
df_game_std = pd.DataFrame(std_scaler.fit_transform(df_game[cols_gaming]), columns=cols_gaming)

#3
target = 'gio_choi'
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(df_game[target], kde=True, ax=axes[0], color='gray').set_title("Original")
sns.histplot(df_game_mm[target], kde=True, ax=axes[1], color='green').set_title("Min-Max Scaled")
sns.histplot(df_game_std[target], kde=True, ax=axes[2], color='red').set_title("Z-Score Scaled")

plt.show()