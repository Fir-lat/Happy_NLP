import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer

# 这里将之前爬取的csv文件转化成为txt文件（和直接读取txt文件效果是一样的）
FILE_PATH = 'E:\大三下\CIS\科研\data1_movie.csv'
df = pd.read_csv(FILE_PATH)
tokenizer = RegexpTokenizer(r'\w+')
stopwords = [['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']]
for w in ['!',',','.','?','','s','\n','\t']:
    stopwords.append(w)
for index,row in df.iterrows():
    total_num = 400
    if (index < total_num) and (type(df.iloc[index,1]) == type('1')):
        # deal with the text
        sentence = []
        for word in tokenizer.tokenize(df.iloc[index,1]):
            if len(word) != 1 and (word.lower() not in stopwords):
                sentence.append(word.lower())
        df.iloc[index,1] = ' '.join(sentence)
        # deal with the label
        if df.iloc[index,0]>=7:
            df.iloc[index,0]=1
        elif df.iloc[index,0]<=4:
            df.iloc[index,0]=0
        else:
            df.iloc[index,0]= np.nan
    else:
        df.iloc[index,0] = np.nan
# delete the one with no distinct tendency in emotion
df = df.dropna(axis=0,how='any')

# 使用text2vec计算similarity
"""
基本的语法就是
from text2vec import Similarity  # 先导入包
sim_model = Similarity()   # 建立模型
score = sim_model.get_score(sentence1,sentence2)  # 直接在模型的get_score()方法，在里面写两个句子就可以知道它们的相似度
# 这个相似度应该是可以
"""
from text2vec import Similarity
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
sentence_similarity  = []
sim_model = Similarity()
for index,row in df.iterrows():
    print(index)
    cusor = index + 1
    if (index < df.shape[0]) and (type(df.iloc[index,1]) == type('1')):
        one_similarity = {}
    while (cusor < df.shape[0]) and (type(df.iloc[cusor,1]) == type('1')):
        score = sim_model.get_score(df.iloc[index,1],df.iloc[cusor,1])
        one_similarity[cusor] = score
        cusor+=1
    sentence_similarity.append(sorted(one_similarity.items(),key = lambda item:item[1],reverse=True))

##保存
fileObject = open('E:/similarity1.txt', 'w')
for ip in sentence_similarity:
    fileObject.write(str(ip))
    fileObject.write('\n')
fileObject.close()
