# split the tweet data into test data and train data
from sklearn.model_selection import train_test_split
import pandas as pd
FILE_PATH = r'E:\大三下\CIS\科研\data2_tweet.csv'  # 设置路径
df = pd.read_csv(FILE_PATH,encoding = "utf-8")   # 读取文件
X = df.text
y = df.airline_sentiment
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.4)
tweet_for_train = {}
tweet_for_train['text'] = X_train
tweet_for_train['label'] = y_train
train_data = pd.DataFrame(tweet_for_train)
tweet_for_test = {}
tweet_for_test['text'] = X_test
tweet_for_test['label'] = y_test
train_data = pd.DataFrame(tweet_for_train)
test_data = pd.DataFrame(tweet_for_test)
train_data.to_csv('E:\大三下\CIS\科研\data2_tweet_train.csv',index=False)
test_data.to_csv('E:\大三下\CIS\科研\data2_tweet_test.csv',index=False)