from gensim.models import FastText
import numpy as np
import pandas as pd

# 미리 훈련된 모델을 로드하거나 자체적으로 훈련시킨 모델을 사용
model = FastText.load('music.model')

# 노래 가사 데이터 로드 (예: CSV 파일)
lyrics_data = pd.read_csv('analyzed_utf8.csv')

# 전처리 함수 정의
def preprocess_text(text):
    # 여기에 텍스트 전처리 단계 추가
    return text

# 전처리 수행
lyrics_data['processed_lyrics'] = lyrics_data['analyzed_lyric'].apply(preprocess_text)

def get_embedding(text, model):
    words = text.split()
    word_embeddings = [model.wv[word] for word in words if word in model.wv]
    if not word_embeddings:
        return np.zeros(model.vector_size)
    return np.mean(word_embeddings, axis=0)

# 노래 가사에 대한 임베딩 생성
lyrics_data['embedding'] = lyrics_data['processed_lyrics'].apply(lambda x: get_embedding(x, model))

# 임베딩을 파일로 저장
lyrics_data.to_pickle('lyrics_embeddings.pkl')

