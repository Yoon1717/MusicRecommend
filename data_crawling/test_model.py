# 모델 테스트
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from gensim.models import FastText
import numpy as np

# 미리 훈련된 모델을 로드하거나 자체적으로 훈련시킨 모델을 사용
model = FastText.load('music.model')
# 노래 가사 데이터 로드 (예: CSV 파일)
lyrics_data = pd.read_csv('analyzed_utf8.csv')

def get_embedding(text, model):
    words = text.split()
    word_embeddings = [model.wv[word] for word in words if word in model.wv]
    if not word_embeddings:
        return np.zeros(model.vector_size)
    return np.mean(word_embeddings, axis=0)


def preprocess_text(text):
    okt = Okt()
    result = []
    d = okt.pos(text, norm=True, stem=True)
    r = []
    for word in d:
        if not word[1] in ['Josa', 'Eomi', 'Punctuation']:
            r.append(word[0])
    # 토큰화된 단어들을 다시 문자열로 결합
    return ' '.join(r)

while True:
    # 사용자 입력
    user_input = input("오늘 일기를 써주세요")
    # 사용자 입력의 임베딩 생성
    user_embedding = get_embedding(preprocess_text(user_input), model)
    # 저장된 임베딩 로드
    lyrics_embeddings = pd.read_pickle('lyrics_embeddings.pkl')
    # 코사인 유사도 계산
    similarity_scores = cosine_similarity([user_embedding], list(lyrics_embeddings['embedding']))[0]
    # 가장 유사한 노래 찾기
    # most_similar_song_index = np.argmax(similarity_scores)
    # most_similar_song = lyrics_data.iloc[most_similar_song_index]
    # print(most_similar_song)
    # 유사도 점수에 따라 내림차순으로 정렬하고 상위 10개 선택
    # 유사도 점수와 함께 원래 데이터프레임을 병합
    lyrics_data['similarity_score'] = similarity_scores
    top_10_songs = lyrics_data.sort_values(by='similarity_score', ascending=False).head(10)
    # 결과 출력
    print(top_10_songs[['title', 'analyzed_lyric', 'similarity_score']])