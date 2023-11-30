import pandas as pd
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
import numpy as np
from gensim.models import fasttext
# # CSV 파일 읽기 (예: 'latin1' 인코딩으로 되어 있다고 가정)
# file_path = 'analyzed.csv'
# df = pd.read_csv(file_path, encoding='cp949')  # 'latin1' 대신 실제 파일 인코딩 사용
#
# # UTF-8 인코딩으로 저장
# output_file_path = 'analyzed_utf8.csv'
# df.to_csv(output_file_path, index=False, encoding='utf-8')

df = pd.read_csv('analyzed_utf8.csv')
df.head()

data = df['analyzed_lyric'].values
okt = Okt()
result = []
for line in data:
    d = okt.pos(line, norm=True, stem=True)
    r = []
    for word in d:
        if not word[1] in ['Josa', 'Eomi', 'Punctuation']:
            r.append(word[0])
    result.append(r)
fasttext_model = fasttext.FastText(result, vector_size=100, window=8, min_count=2, sg=1)
fasttext_model.save('music.model')

while True:
    embedding_matrix = np.zeros((100, 100))
    docs = input("검색 문장").split()
    token = Tokenizer()
    token.fit_on_texts(docs)
    for word, idx in token.word_index.items():
        if word in fasttext_model.wv:
            embedding_matrix[idx] = fasttext_model.wv[word]
    print(embedding_matrix)
    # print('fasttext:', fasttext_model.wv.most_similar(positive=[text]))