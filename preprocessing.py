import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\DaTaProcessing\1\ITA105_Lab_1.csv")
df.head()
print(df.head(),"\n")
#  1Khám phá dữ liệu
# - Kiểm tra kích thước dữ liệu (số dòng và cột).
# - Xem thống kê mô tả của các cột số.
# - Kiểm tra giá trị thiếu trong các cột.
df.shape
print("1-Kich thuoc du lieu : ",df.shape,"(dong,cot)\n")

print("2-Kieu du lieu va gia tri thieu :\n")
print(df.info(),"\n")

print("3-Thong ke so lieu mo ta cac cot so\n")
print(df.describe(),"\n")

#  2
# Xử lý dữ liệu thiếu
# - Dùng .isnull().sum() để phát hiện giá trị thiếu.
# - Điền giá trị thiếu với mean/median/mode.
# - So sánh kết quả với phương pháp dropna()
print("4-Kiem tra gia tri thieu-Dem so luong NaN theo cot")
print(df.isnull().sum(),"\n")
print("Kiem tra cu the mot cot :")
cols=["ProductID","Price","StockQuantity","Rating"]
for i in cols:
    df.isnull().sum()
    print("Cot",i,"thieu",df[i].isnull().sum())
print("\n")

print("Dien gia tri bang mean/median/mode")
df["Price_mean"] = df["Price"].fillna(df["Price"].mean(),inplace=False)
print(df["Price_mean"],"\n")

df["Rating_median"] = df["Rating"].fillna(df["Rating"].median(),inplace=False)
print(df["Rating_median"],"\n")

df["StockQuantity_mode"] = df["StockQuantity"].fillna(df["StockQuantity"].mode()[0],inplace=True)
print(df["StockQuantity"],"\n")

print("Loại bo dong - dropna")
df_dropped = df.dropna()
# so sanh Fillna vs Dropna
# Fillna : Ghi lai du lieu
# Dropna : Xoa dong

# 3
# Xử lý dữ liệu lỗi
# - Kiểm tra và xử lý các giá trị bất hợp lý trong cột Price và
# StockQuantity.
# - Lọc các giá trị không hợp lệ trong cột Rating.
print(" Kiem tra va xu ly cac gia tri bat hop ly trong cot Price va StockQuantity")
df[df["Price"]<0]
df = df[df["Price"]>=0]
df = df[df["StockQuantity"] >= 0]

print("Loc cac gia tri khong hop le trong cot Rating")
df = df[(df["Rating"] > 1) & (df["Rating"] <= 5)]

# 4
# Làm mượt dữ liệu nhiễu
# - Áp dụng Moving Average để làm mượt dữ liệu cột Price.
# - Vẽ biểu đồ line trước và sau khi làm mượt.
df["Price_smooth"] = df["Price"].rolling(3).mean()

plt.plot(df["Price"], label="Original Price")
plt.plot(df["Price_smooth"], label = "Smoothed Price")
plt.legend()
plt.show()

# 5
# Chuẩn hóa dữ liệu
# - Chuyển tất cả giá trị trong cột Category thành chữ thường.
# - Loại bỏ ký tự thừa trong cột Description.
# - Chuyển đổi đơn vị giá từ USD sang VND.

df["Category"] = df["Category"].str.lower()

df["Description"] = df["Description"].str.strip()

df["Price_VND"] = df [" Price"] * 26500