import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer

FILE_PATH = 'IMDB Dataset.csv'
df = pd.read_csv(FILE_PATH)
tokenizer = RegexpTokenizer(r'\w+')
stopwords = []
for w in ['!',',','.','?','-s','-ly','','s','\n','\t']:
    stopwords.append(w)
for index,row in df.iterrows():
    print(index)
    if (index < df.shape[0]) and type(df.iloc[index,0]) == type('1'):
        # deal with the text
        sentence = []
        for word in tokenizer.tokenize(df.iloc[index,0]):
            if len(word) != 1 and (word.lower() not in stopwords):
                sentence.append(word.lower())
        df.iloc[index,0] = ' '.join(sentence)
        # deal with the label
        if df.iloc[index,1] == 'positive':
            df.iloc[index,1]=1
        elif df.iloc[index,1] == 'negative':
            df.iloc[index,1]=0
        else:
            df.iloc[index,1]= np.nan
# delete the one with no distinct tendency in emotion
df = df.dropna(axis=0,how='any')
df.to_csv('IMDB Dataset.txt', sep='\t', index=False,header = False)