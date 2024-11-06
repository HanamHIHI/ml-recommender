import csv
from sentence_transformers import util
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
from tqdm import tqdm 
import time
import umap

tags = ['hanam']
tag = tags[0]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_count = 800 # hyper_params.
test_count = 200
batch_size = 16
epochs = 128

mapper_df = pd.read_csv("hanam_dict.csv", encoding="utf-8")
mapper = list(mapper_df['1'])

# f = open('preprocessed_urls_hanam_restaurant_real_url_review.csv','r', encoding='utf-8') # *.data.tsv 파일은 강의평, 강의 인덱스 데이터 파일
# rdr = csv.reader(f, delimiter='\t')
# test_data = [[] for _ in list(range(len(mapper)))]
# temp_data = []
# for row in rdr:
#     test_data[int(row[1])].append(str(row[0]))
#     temp_data.append(row)
# f.close()

data_df = pd.read_csv("preprocessed_urls_hanam_restaurant_real_url_review.csv", encoding="utf-8")
test_data = [[] for _ in list(range(len(mapper)))]
temp_data = []

for idx in range(len(data_df)):
    test_data[mapper.index(str(data_df.iloc[idx]["name"]))].append(str(data_df.iloc[idx]["review"]))
    temp_data.append([str(data_df.iloc[idx]["review"]), mapper.index(str(data_df.iloc[idx]["name"]))])

test_examples = []
f = open(tag + '_test_data_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
rdr = csv.reader(f)
for row in rdr:
    test_examples.append(row)
f.close()

model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS', device=device) # pre-trained 모델 불러오기
model.eval() # gpu 너무 많이 잡아 먹어서 달아놓은 코드

import os
from collections import OrderedDict

try:
    model_state_dict = torch.load('basic_model_'+str(train_count+test_count)+'.pt', map_location=device)
except FileNotFoundError:
    model_state_dict = torch.load('basic_model_'+str(train_count+test_count)+'.pt', map_location=device)

try:
    model.load_state_dict(model_state_dict)
except RuntimeError:
    model.load_state_dict(model_state_dict, strict=False)

import numpy as np

vectors = [[] for _ in list(range(len(mapper)))]
try:
    f = open(tag + '_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(float(_row))
            # if(i == 0):
            #     print(_row, float(_row), 0.3396976888179779, "int(temp_data[i][1])", int(temp_data[i][1]))
        vectors[int(temp_data[i][1])].append((np.array(floatCastedRow), int(temp_data[i][1])))
    f.close()

except FileNotFoundError:
    f = open(tag + '_vectors_' + str(train_count+test_count) + '.csv','w', encoding='utf-8', newline='')
    writer = csv.writer(f)

    for i in tqdm(list(range(len(mapper)))): # 각 강의평마다 주어진 강의평 벡터의 평균을 각 강의와 매핑합니다.
        for j in list(range(len(test_data[i]))):
            vector = model.encode(test_data[i][j])
            listedVector = vector.tolist()
            writer.writerow(listedVector)
        print(i, "encoding done")

    f.close()

    f = open(tag + '_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(float(_row))
            # if(i == 0):
            #     print(_row, float(_row))
        vectors[int(temp_data[i][1])].append((np.array(floatCastedRow), int(temp_data[i][1])))
    f.close()

dimList = []
vectorData = [[] for _ in list(range(len(mapper)))]
for i in tqdm(list(range(len(vectors)))):
    for j in list(range(len(vectors[i]))):
        vectorData[i].append(vectors[i][j][0])
        dimList.append(np.array(vectors[i][j][0]).shape)

print("Start mean cal-ing")
start_time = time.time()

mean_vectors = []
try:
    f = open(tag + '_mean_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(np.double(_row))
            # if(i == 0):
            #     print(_row, float(_row), 0.3396976888179779, "int(temp_data[i][1])", int(temp_data[i][1]))
        mean_vectors.append([np.array(floatCastedRow), i])
    f.close()

    print("reading mean vectors complete.")
except FileNotFoundError:
    f = open(tag + '_mean_vectors_' + str(train_count+test_count) + '.csv','w', encoding='utf-8', newline='')
    writer = csv.writer(f)

    for i in tqdm(list(range(len(mapper)))):
        vector = model.encode(test_data[i])
        vector = np.mean(vector, axis=0)

        listedVector = vector.tolist()
        writer.writerow(listedVector)
    f.close()

    f = open(tag + '_mean_vectors_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    rdr = csv.reader(f)
    for i, row in tqdm(enumerate(rdr)):
        floatCastedRow = []
        for _row in row:
            floatCastedRow.append(np.double(_row))
            # if(i == 0):
            #     print(_row, float(_row))
        mean_vectors.append([np.array(floatCastedRow), i])
    f.close()

print(f"mean cal-ing done after {time.time() - start_time:.2f} sec")

print("Start recommending")
start_time = time.time()

ndarray_vectors = np.array(list(zip(*mean_vectors))[0])
# pca = PCA(n_components=16)
# pca.fit(ndarray_vectors)
# compressed_vectors = pca.transform(ndarray_vectors)

reducer = umap.UMAP()
reducer.fit(ndarray_vectors)
print("fitting complete.")
compressed_vectors = reducer.transform(ndarray_vectors)
print("transforming complete.")

print(ndarray_vectors.shape, compressed_vectors.shape)
# print(reducted_vectors[0])
# print(reducted_vectors[1])

# Two parameters to tune:
# min_cluster_size: Only consider cluster that have at least 25 elements
# threshold: Consider sentence pairs with a cosine-similarity larger than threshold as similar
clusters = util.community_detection(np.array(compressed_vectors), min_community_size=4, threshold=0.1)

clusterd_compressed_vectors = [[] for i in range(len(clusters))]
compressed_mean_vectors = []
# Print for all clusters the top 3 and bottom 3 elements
for i, cluster in enumerate(clusters):
    print(f"\nCluster {i + 1}, #{len(cluster)} Elements ")
    # for sentence_id in cluster[0:1]:
    #     print("\t", sentence_id, mapper[sentence_id])
    # print("\t", "...")
    # for sentence_id in cluster[-1:]:
    #     print("\t", sentence_id, mapper[sentence_id])

    for sentence_id in cluster:
        clusterd_compressed_vectors[i].append((compressed_vectors[sentence_id], sentence_id))
    # list(zip(*a))[0]
    compressed_mean_vector = np.mean(list(zip(*clusterd_compressed_vectors[i]))[0], axis=0)
    compressed_mean_vectors.append(compressed_mean_vector)

targetText = "비가 올 때 가면 좋아요" #상상 강의평
targetVector = reducer.transform(model.encode([targetText])) # targetVector는 데스트 할 text string의 sentence vectorpre_results = []

pre_results = []
pre_answerList = []
for j in list(range(len(compressed_mean_vectors))):
    if(j == 0):
        print(compressed_mean_vectors[j])
        print("-----------------------------")
        print(targetVector)
        print("-----------------------------")

    similarities = util.cos_sim(compressed_mean_vectors[j], targetVector) # compute similarity between sentence vectors
    pre_results.append((j, j, float(similarities)))
pre_results.sort(key = lambda x : -x[2])
pre_answer = pre_results[0][0]

results = []
answerList = []
for j in list(range(len(clusterd_compressed_vectors[pre_answer]))):
    similarities = util.cos_sim(clusterd_compressed_vectors[pre_answer][j][0], targetVector) # compute similarity between sentence vectors
    results.append((j, mapper[clusterd_compressed_vectors[pre_answer][j][1]], float(similarities)))
results.sort(key = lambda x : -x[2])

print(targetText, "에 적합한 식당은")
print("번호" + " " + "상호명" + " " + "score")
print("="*45)
for result in results[:10]:
    printedString = result
    print(printedString)
print("입니다.")

print(f"Recommending done after {time.time() - start_time:.2f} sec")