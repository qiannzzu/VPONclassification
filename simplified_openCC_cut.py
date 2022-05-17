#encoding=utf-8  
import csv
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import jieba.posseg as pseg
import re
import gensim
import nltk
nltk.download('punkt')
from nltk import word_tokenize
from gensim.models import word2vec
from scipy.sparse import coo_matrix
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import *
from sklearn.preprocessing import LabelEncoder
from opencc import OpenCC

#導入資料集
item = pd.read_csv('./input/items_x_complete_columns.csv')
cc = OpenCC('tw2sp')
for i in range(0,len(item)):
    item['item_name'][i] = cc.convert(item['item_name'][i])
    item['item_brand'][i] = cc.convert(item['item_brand'][i])
    item['item_category'][i] = cc.convert(item['item_category'][i])
#新增自定義詞典和停用詞典
jieba.load_userdict("./input/itemBrand_simplified.txt") #自定義詞典設為所有brand name 但造成accuracy下降 故不採用
stop_list = pd.read_csv('./input/hit_stopword.txt',
                        engine='python',
                        encoding='UTF-8',
                        error_bad_lines=False,
                        delimiter="\r\n",
                        names=['t'])['t'].tolist()


#中文分詞函式
def txt_cut(juzi):
    return [w for w in jieba.lcut(juzi) if w not in stop_list]

#寫入分詞結果
fw = open('fenci_data_simplified_openCC.csv', "a+", newline = '',encoding = 'gb18030')
writer = csv.writer(fw)  
writer.writerow(['item_name','item_category'])

#label encoding
lbl = preprocessing.LabelEncoder() #將文字進行資料前處理->Label encoding
item['item_category'] = lbl.fit_transform(item['item_category']) #item_category欄位做Label encoding

#文字資料預處理

 
# # 只提取出中文出來 準確率較低 捨棄
# for i in range(0,len(item)): 
#   new_data = re.findall('[\u4e00-\u9fa5]+', item['item_name'][i], re.S) 
#   item['item_name'][i] = "".join(new_data) 

# 去除數字 準確度較高
for i in range(0,len(item)):
  input_str = item['item_name'][i]
  item['item_name'][i] = re.sub(r'\d+', '', input_str) 
  # print(result) 

#斷詞
# 使用csv.DictReader讀取檔案中的資訊
labels = []
contents = []
outputxt = open('corpusSegDone_simplified_openCC.txt', 'w', encoding='utf-8')
for i in range(0,len(item)):
    res = item['item_category'][i]
    labels.append(res)
    content = item['item_name'][i]
    a=str(content)
    seglist = txt_cut(a) #含去除stoplist
    output = ' '.join(list(seglist))   #原本每列都被分隔['','','','']->改由空格拼接 :Dr AV NP 三洋 洗衣 機專用 濾網 超值 四入 組
    #加上品牌之column做為特徵
    output = output+' '+item['item_brand'][i]
    #加上價錢之column做為特徵(使level1之accuracy降低 捨棄不看)
    # output = output+' '+str(item['price'][i])

    contents.append(output)
    outputxt.write(output)
    outputxt.write('\n')
    
    #檔案寫入fenci.csv
    tlist = []
    tlist.append(output)
    tlist.append(res)
    writer.writerow(tlist)
outputxt.close()
def multiclass_logloss(actual, predicted, eps=1e-15): #在kaggle上常用於計算分類問題之Loss func
    """對數損失度量（Logarithmic Loss  Metric）的多分類版本。
    :param actual: 包含actual target classes的陣列
    :param predicted: 分類預測結果矩陣, 每個類別都有一個概率
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
#----------------------------------資料處理--------------------------------
#將文字中的詞語轉換為詞頻矩陣 矩陣元素a[i][j] 表示j詞在i類文字下的詞頻
vectorizer = CountVectorizer(min_df=5) #min_df=5: 如果某個詞的document frequence小於min_df，則這個詞不會被當作關鍵詞。
#該類會統計每個詞語的tf-idf權值
transformer = TfidfTransformer()

#第一個fit_transform是計算tf-idf 第二個fit_transform是將文字轉為詞頻矩陣
tfidf = transformer.fit_transform(vectorizer.fit_transform(contents))

# 獲取詞袋模型中的所有詞語(可用於查看有哪些詞)
# word = vectorizer.get_feature_names()
# print("單詞數量:", len(word))

#將tf-idf矩陣抽取出來，元素w[i][j]表示j詞在i類文字中的tf-idf權重
#X = tfidf.toarray()
X = coo_matrix(tfidf, dtype=np.float32).toarray() #稀疏矩陣 注意float

#----------------------------------資料分配--------------------------------
#資料集做split (for tfidf)
#使用 train_test_split 分割 X y 列表 train/test: 0.8/0.2
X_train, X_test,\
y_train, y_test = train_test_split(X, labels, 
                   test_size=0.2, 
                   random_state=1)
                   # 邏輯迴歸分類方法模型
LR = LogisticRegression(solver='lbfgs', max_iter=400)
LR.fit(X_train, y_train)
print('Tfidf+LR模型 準確度:{}'.format(LR.score(X_test, y_test)))
predictions = LR.predict_proba(X_test) 
arr = np.array(y_test) # converting list to array
print ("Tfidf+LR logloss: %0.3f " % multiclass_logloss(arr, predictions))
'''
# model = gensim.models.Word2Vec(contents, size=100, min_count=5) #分詞結果轉為word2vec詞向量模型（100維）
sentences = word2vec.LineSentence("corpusSegDone_simplified_openCC.txt")
model = word2vec.Word2Vec(sentences, vector_size=100)
embeddings_index = dict(zip(model.wv.index_to_key, model.wv.vectors ))  #Word2Vec模型中的词汇表存储在model.wv.index2word 特征向量存储在叫做syn0的numpy数组
#資料集做split (for word2vec)
xtrain, xvalid, ytrain, yvalid = train_test_split(contents, labels,
                                                  random_state=1,
                                                  test_size=0.2, shuffle=True)
#該函式會將語句轉化為一個標準化的向量（Normalized Vector）
def sent2vec(s):
    # print (s)
    words = word_tokenize(s) #斷詞功能 ex"At eight o'clock on Thursday morning."->['At','eight',"o'clock",...]
    words = [w for w in words if w.isalpha()]
    M = []

    for w in words:

        try:
            # print(embeddings_index[w].shape)
            M.append(embeddings_index[w])

        except:
            continue

    M = np.array(M)
    v = M.sum(axis=0)
    # print("\nv.shape:",v.shape)

    if type(v) != np.ndarray:
        # print("\n!")
        return np.zeros(100)

    return v / np.sqrt((v ** 2).sum())
# contents向量化
from tqdm import tqdm #顯示進度條
xtrain_w2v  = [sent2vec(x) for x in tqdm(xtrain)]
# xtrain_w2v  = sent2vec(xtrain[2])
xvalid_w2v  = [sent2vec(x) for x in tqdm(xvalid)]


#使用Cross Entropy(又稱logloss)是最常使用於分類問題的損失函數(loss functions)
def multiclass_logloss(actual, predicted, eps=1e-15): #在kaggle上常用於計算分類問題之Loss func
    """對數損失度量（Logarithmic Loss  Metric）的多分類版本。
    :param actual: 包含actual target classes的陣列
    :param predicted: 分類預測結果矩陣, 每個類別都有一個概率
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

LR_w2v = LogisticRegression(solver='lbfgs', max_iter=400)
LR_w2v.fit(xtrain_w2v, ytrain)
predictions = LR_w2v.predict_proba(xvalid_w2v) 
print('LR_w2v模型的準確度:{}'.format(LR_w2v.score(xvalid_w2v, yvalid)))
arr = np.array(yvalid) # converting list to array
print ("LR_w2v logloss: %0.3f " % multiclass_logloss(arr, predictions))
'''