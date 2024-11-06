import csv
import pandas as pd

# f = open('preprocessed_urls_hanam_restaurant_real_url_review.csv','r', encoding='utf-8') # *.data.tsv 파일은 강의평, 강의 인덱스 데이터 파일
# rdr = csv.reader(f)
# data = []
# for row in rdr:
#     data.append(row)
# f.close()

df = pd.read_csv("preprocessed_urls_hanam_restaurant_real_url_review.csv")

print(len(df))

name_list = df["name"]
unique_name_list = list(set(name_list))
name_dict = []

for i in range(len(unique_name_list)):
    name_dict.append([i, unique_name_list[i]])

# for idx in range(10):
#     print(name_dict[idx])

df2 = pd.DataFrame(name_dict)
df2.to_csv("hanam_dict.csv")