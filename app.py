from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from gensim.models import FastText
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# 미리 훈련된 모델을 로드하거나 자체적으로 훈련시킨 모델을 사용
model = FastText.load('music.model')
# 노래 가사 데이터 로드 (예: CSV 파일)

lyrics_data = pd.read_csv('last_merged_data.csv')
genres_list = []
genre_rate = []
possible_genres = ['GN0100', 'GN0200', 'GN0300', 'GN0400', 'GN0500', 'GN0600', 'GN0700', 'GN0800']

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


def remove_duple_top_songs(num, content):
    # 사용자 입력의 임베딩 생성
    user_embedding = get_embedding(preprocess_text(content), model)
    # 저장된 임베딩 로드
    lyrics_embeddings = pd.read_pickle('lyrics_embeddings.pkl')
    # 코사인 유사도 계산
    similarity_scores = cosine_similarity([user_embedding], list(lyrics_embeddings['embedding']))[0]
    # 유사도 점수에 따라 내림차순으로 정렬
    lyrics_data['similarity_score'] = similarity_scores
    sorted_songs = lyrics_data.sort_values(by='similarity_score', ascending=False)
    # 중복 song_no를 제거하고 상위 10개의 곡 선택
    top_songs = []
    seen_song_no = set()  # 이미 선택한 song_no를 추적하기 위한 집합
    for _, row in sorted_songs.iterrows():
        song_no = row['song_no']
        if song_no not in seen_song_no:
            top_songs.append(row.to_dict())
            seen_song_no.add(song_no)
            if len(top_songs) == num:
                break

    return top_songs
def get_genre_rate(list):
    genre_rate = []  # 결과를 담을 리스트 초기화
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
    return genre_rate


def make_chart(rate):
    genre_names = {
        'GN0100': 'Ballade',
        'GN0200': 'Dance',
        'GN0300': 'Rap/Hiphop',
        'GN0400': 'R&B/Soul',
        'GN0500': 'Indie',
        'GN0600': 'Rock/Metal',
        'GN0700': 'Trot',
        'GN0800': 'Folk/Bluse',
    }

    # 장르 코드를 장르명으로 변환
    for item in rate:
        item['Genre'] = genre_names.get(item['Genre'], item['Genre'])

    # 'Genre'와 'Ratio'를 추출하여 리스트로 저장
    genre_list = [item['Genre'] for item in rate]
    ratio_list = [item['Ratio'] * 100 for item in rate]

    # categories에 'Genre' 추가
    categories = [*genre_list, genre_list[0]]

    # grade1에 'Ratio' 추가
    grade1 = [*ratio_list, ratio_list[0]]
    # 나머지 그래프 그리는 코드
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(grade1))
    plt.figure(figsize=(8, 8))
    ax = plt.subplot(polar=True)
    plt.xticks(label_loc, labels=categories, fontsize=13)
    ax.plot(label_loc, grade1, label='my mode', linestyle='dashed', color='lightcoral')
    ax.fill(label_loc, grade1, color='lightcoral', alpha=0.3)
    ax.legend()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = base64.b64encode(img_buffer.read()).decode()

    # 이미지 데이터를 HTML 페이지로 전달
    return img_data



@app.route("/", methods=["GET", "POST"])
def index():
    # top_10_songs = []  # 빈 리스트 생성
    # top_50_songs = []  # 빈 리스트 생성

    if request.method == "POST":
        # 폼 데이터를 처리하고 원하는 동작을 수행하세요
        content = request.form["content"]
        top_10_songs = remove_duple_top_songs(10, content)
        top_50_songs = get_top_songs(50, content)
        gen_rate = get_genre_rate(top_50_songs)
        img_data = make_chart(gen_rate)

        return render_template("store.html"
                                       , top_10_songs=top_10_songs
                                       , top_50_songs=top_50_songs
                                       , gen_rate=gen_rate
                                       , img_data=img_data)  # top_songs 변수를 store.html로 전달
    else:
        return render_template("index.html")  # top_songs

@app.route("/store", methods=["GET", "POST"])
def store():
    # 다른 페이지(예: store.html)의 뷰 코드
    return render_template("store.html")

if __name__ == "__main__":
    app.run(debug=True, host="192.168.0.21")