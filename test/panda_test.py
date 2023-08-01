import pandas as pd
import numpy as np 
 
grades = {
    "name": ["Mike", "Sherry", "Cindy", "John"],
    "math": [80, 75, 93, 86],
    "chinese": [63, 90, 85, 70]
}
 
df = pd.DataFrame(grades)
#df.index = ["s1", "s2", "s3", "s4"]  #自訂索引值
#df.columns = ["student_name", "math_score", "chinese_score"]  #自訂欄位名稱

print("使用字典來建立df：")
print(df)
 
print("=================================")
''' 
grades = [
    ["Mike", 80, 63],
    ["Sherry", 75, 90],
    ["Cindy", 93, 85],
    ["John", 86, 70]
]
 
new_df = pd.DataFrame(grades)
 
print("使用陣列來建立df：")
print(new_df)

print("=================================")
'''
'''
load_df = df.head(2)
print("取得最前面的兩筆資料")
print(load_df)

load2_df = df.tail(3)
print("取得最後面的三筆資料")
print(load2_df)
'''
'''
print("取得單一欄位資料(型別為Series)")
print(df["name"])

print("=================================")
 
print("取得單一欄位資料(型別為DataFrame)")
print(df[["name"]])
 
print("=================================")
 
print("取得多欄位資料(型別為DataFrame)")
print(df[["name", "chinese"]])
 
print("=================================")
 
print("取得索引值0~2的資料")
print(df[0:3])
print("=================================")
'''
'''
print("利用at()方法取得索引值為1的math欄位資料")
print(df.at[1, "math"])
print("=================================")

print("利用iat()方法取得索引值為1的第一個欄位資料")
print(df.iat[1, 0])
print("=================================")

print("取得資料索引值為1和3的name及chinese欄位資料集")
print(df.loc[[1, 3], ["name", "chinese"]])
print("=================================")

print("取得資料索引值為1和3的第一個及第三個欄位資料集")
print(df.iloc[[1, 3], [0, 2]])
print("=================================")
'''
'''
df.insert(2, column="engilsh", value=[88, 72, 74, 98])
print("在第三欄的地方新增一個欄位資料")
print(df)
print("=================================")

new2_df = df.append({
    "name": "Henry",
    "math": 60,
    "chinese": 62
}, ignore_index=True)
 
print("新增一筆資料")
print(new2_df)
print("=================================")

df2 = pd.DataFrame({
    "name": ["Henry"],
    "math": [60],
    "chinese": [62]
})
 
new3_df = pd.concat([df, df2], ignore_index=True)
print("合併df來新增資料")
print(new3_df)
print("=================================")
'''
'''
df.at[1, "math"] = 100  #修改索引值為1的math欄位資料
df.iat[1, 0] = "Larry"  #修改索引值為1的第一個欄位資料
print("修改後的df")
print(df)
print("=================================")

new4_df = df.drop(["math"], axis=1)
print("刪除math欄位")
print(new4_df)
print("=================================")

new5_df = df.drop([0, 3], axis=0)  # 刪除第一筆及第四筆資料
print("刪除第一筆及第四筆資料")
print(new5_df)
print("=================================")
'''
'''
grades = {
    "name": ["Mike", "Sherry", np.NaN, "John"],
    "city": ["Taipei", np.NaN, "Kaohsiung", "Taichung"],
    "math": [80, 75, 93, 86],
    "chinese": [63, 90, 85, 70]
}
 
df = pd.DataFrame(grades)
print("原來的df")
print(df)
 
print("======================================")
 
new6_df = df.dropna()
print("刪除空值後的df")
print(new6_df)
print("======================================")
'''

''' 
grades = {
    "name": ["Mike", "Mike", "Cindy", "John"],
    "city": ["Taipei", "Taipei", "Kaohsiung", "Taichung"],
    "math": [80, 80, 93, 86],
    "chinese": [80, 80, 93, 86]
}
 
df = pd.DataFrame(grades)
print("原來的df")
print(df)

print("======================================")
 
new7_df = df.drop_duplicates()
print("刪除重複值後的df")
print(new7_df)
print("======================================")
'''
'''
grades = {
    "name": ["Mike", "Sherry", "Cindy", "John"],
    "math": [80, 75, 93, 86],
    "chinese": [63, 90, 85, 70]
}
 
df = pd.DataFrame(grades)
 
print("原來的df")
print(df)
 
print("=================================")
 
print("篩選math大於80的資料集")
print(df[df["math"] > 80])
print("=================================")
'''
'''
grades = {
    "name": ["Mike", "Sherry", "Cindy", "John"],
    "math": [80, 75, 93, 86],
    "chinese": [63, 90, 85, 70]
}
 
df = pd.DataFrame(grades)
 
print("原來的df")
print(df)
 
print("=================================")
 
print("篩選name欄位包含John的資料集")
print(df[df["name"].isin(["John"])])
print("=================================")
'''
'''
grades = {
    "name": ["Mike", "Sherry", "Cindy", "John"],
    "math": [80, 75, 93, 86],
    "chinese": [63, 90, 85, 70]
}
 
df = pd.DataFrame(grades)
df.index = ["s3", "s1", "s4", "s2"]  # 自訂資料索引值
 
print("原來的df")
print(df)
 
print("============================")
 
new_df = df.sort_index(ascending=True)
print("遞增排序")
print(new_df)
 
print("============================")
 
new8_df = df.sort_index(ascending=False)
print("遞減排序")
print(new8_df)
print("=================================")
'''
'''
grades = {
    "name": ["Mike", "Sherry", "Cindy", "John"],
    "math": [80, 75, 93, 86],
    "chinese": [63, 90, 85, 70]
}
 
df = pd.DataFrame(grades)
 
print("原來的df")
print(df)
 
print("============================")
 
new_df = df.sort_values(["math"], ascending=True)
print("遞增排序")
print(new_df)
 
print("============================")
 
new9_df = df.sort_values(["math"], ascending=False)
print("遞減排序")
print(new9_df)
print("=================================")
'''

df.to_csv("test.csv")
print("=================================")

df = pd.read_csv('test.csv')
print(df)

