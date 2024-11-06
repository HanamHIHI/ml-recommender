import csv
import pandas as pd

tags = ['hanam']

for tag in tags:
    # f = open('hanam_dict.csv','r', encoding='utf-8') # *.dict.tsv 파일은 강의 인덱스 - 강의 이름+교수명 매핑 파일
    # rdr = csv.reader(f)
    # mapper = []
    # for row in rdr:
    #     mapper.append(row[2])
    # f.close()

    mapper_df = pd.read_csv("hanam_dict.csv", encoding="utf-8")
    mapper = list(mapper_df['1'])
    # print(mapper)

    # f = open('preprocessed_urls_hanam_restaurant_real_url_review.csv','r', encoding='utf-8') # *.data.tsv 파일은 강의평, 강의 인덱스 데이터 파일
    # rdr = csv.reader(f)
    # data = []
    # for row in rdr:
    #     data.append(row)
    # f.close()

    data_df = pd.read_csv("preprocessed_urls_hanam_restaurant_real_url_review.csv", encoding="utf-8")
    data = data_df[['Unnamed: 0', 'name', 'rate', 'review']].values.tolist()
    # print(data)

    import random

    train_examples = []
    test_examples = []
    train_count = 800 # hyper_params.
    test_count = 200
    # batch_size = 16
    # epochs = 32

    # print(data[0])
    # print(data[1])

    preTrain = open(tag + '_preTrain_data_' + str(train_count+test_count) + '.csv','w', encoding='utf-8')
    preTest = open(tag + '_predata_' + str(train_count+test_count) + '.csv','w', encoding='utf-8')
    for k in list(range(len(data))):
        try:
            if(k % 5 != 0):
                preTrain.write(data[k][-1].replace(',','') + ',' + data[k][1] + '\n')
            else:
                preTest.write(data[k][-1].replace(',','') + ',' + data[k][1] + '\n')
        except AttributeError:
            if(k % 5 != 0):
                preTrain.write('' + ',' + data[k][1] + '\n')
            else:
                preTest.write('' + ',' + data[k][1] + '\n')

    preTrain.close()
    preTest.close()

    preTrain = open(tag + '_preTrain_data_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    preTrainRdr = csv.reader(preTrain)
    preTest = open(tag + '_predata_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    preTestRdr = csv.reader(preTest)

    train_data = [[] for _ in list(range(len(mapper)))]
    test_data = []
    for row in preTrainRdr:
        train_data[mapper.index(row[1])].append(str(row[0]))
    for row in preTestRdr:
        test_data.append(row)

    preTrain.close()
    preTest.close()

    print(train_data[0])
    for idx, _ in enumerate(train_data[0]):
        print(idx, _)
    # print(test_data[0])

    for k in range(train_count): # training data 생성. (강의A_강의평1, 강의A_강의평2, 강의B_강의평1) 형식의 튜플을 train_example에 저장
        i = -1
        while(i<0 or len(train_data[i]) == 0):
            i = random.randint(0, len(train_data)-1)
        j = -1
        while(j <0 or j == i or len(train_data[j]) == 0):
            j = random.randint(0, len(train_data)-1)

        train_examples.append([train_data[i][random.randint(0, len(train_data[i])-1)], train_data[i][random.randint(0, len(train_data[i])-1)], train_data[j][random.randint(0, len(train_data[j])-1)]])

    print(data[0])

    randomList = []
    k = 0
    while(k < test_count):
        l = random.randint(0, len(data)-1)

        if(l in randomList):
            continue
        else:
            test_examples.append([data[l][-1], data[l][1]])
            k = k+1

    print(train_examples[0])
    print(test_examples[0])

    f = open(tag + '_train_data_' + str(train_count+test_count) + '.csv','w', encoding='utf-8')
    for train_example in train_examples:
        f.write(train_example[0].replace(',','') + ',' + train_example[1].replace(',','') + ',' + train_example[2].replace(',','') + '\n')
    f.close()

    f = open(tag + '_test_data_' + str(train_count+test_count) + '.csv','w', encoding='utf-8')
    for test_example in test_examples:
        try:
            f.write(test_example[0].replace(',','') + ',' + test_example[1] + '\n')
        except AttributeError:
            f.write('' + ',' + test_example[1] + '\n')
    f.close()
