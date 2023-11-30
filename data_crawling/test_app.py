
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt



# 미리 훈련된 모델을 로드하거나 자체적으로 훈련시킨 모델을 사용
model = FastText.load('music.model')
# 노래 가사 데이터 로드 (예: CSV 파일)
lyrics_data = pd.read_csv('analyzed_utf8.csv')
genres_list = []
genre_rate = []
possible_genres = ['GN0100', 'GN0200', 'GN0300', 'GN0400', 'GN0500', 'GN0600', 'GN0700', 'GN0800']
content = '파업해'

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

def get_top_songs(num, content):
    # 사용자 입력의 임베딩 생성
    user_embedding = get_embedding(preprocess_text(content), model)
    # 저장된 임베딩 로드
    lyrics_embeddings = pd.read_pickle('lyrics_embeddings.pkl')
    # 코사인 유사도 계산
    similarity_scores = cosine_similarity([user_embedding], list(lyrics_embeddings['embedding']))[0]
    # 유사도 점수에 따라 내림차순으로 정렬하고 상위 num 개 선택
    lyrics_data['similarity_score'] = similarity_scores
    top_songs = lyrics_data.sort_values(by='similarity_score', ascending=False).head(num).to_dict('records')
    return top_songs

top_10_songs = []  # 빈 리스트 생성
top_50_songs = []  # 빈 리스트 생성
top_10_songs = get_top_songs(10, content)
top_50_songs = get_top_songs(50, content)
print(top_50_songs)
print(type(top_50_songs))

def get_genre_list(list):
    for song_info in list:
        genre = song_info['genre']
        genres_list.append(genre)
    print(genres_list)
    # 각 장르의 갯수 계산
    genre_counts = Counter(genres_list)
    # 전체 곡 수
    total_songs = len(genres_list)
    # 각 장르의 비율 계산
    genre_ratios = {genre: count / total_songs for genre, count in genre_counts.items()}
    # 결과 출력
    for genre in possible_genres:
        count = genre_counts.get(genre, 0)
        ratio = round(genre_ratios.get(genre, 0.0),2)
        print(f'Genre: {genre}, Count: {count}, Ratio: {ratio:.2%}')
        genre_rate.append({'Genre': genre, 'Count': count, 'Ratio': ratio } )

get_genre_list(top_50_songs)

print('rate:', genre_rate)

# 카테고리와 해당 비율 추출
categories = [item['Genre'] for item in genre_rate]
ratios = [item['Ratio'] * 100 for item in genre_rate]

# 각 카테고리의 각도 계산
# angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
# angles += angles[:1]
# 각 카테고리의 각도 계산
angles = np.linspace(0, 2 * np.pi, len(possible_genres), endpoint=False).tolist()
angles += angles[:1]


# 그래프 그리기
fig, ax = plt.subplots(subplot_kw=dict(polar=True))

# 각 카테고리에 대한 데이터 플로팅
ax.fill(angles, ratios, alpha=0.25)

# 각 카테고리 레이블 표시
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)

# 그래프 최대치를 100으로 설정
ax.set_ylim(0, 100)

# 그래프를 화면에 표시
plt.show()





