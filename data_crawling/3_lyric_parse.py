import pandas as pd
from konlpy.tag import Okt

# 엑셀 파일 로드
df = pd.read_excel("song_detail.xlsx", engine="openpyxl")
titles = df["title"]
lyrics = df["lyric"]
keyword = df["keyword"]
song_no = df["song_no"]

# titles, keyword, song_no 열을 리스트로 변환
titles_list = titles.tolist()
keyword_list = keyword.tolist()
song_no_list = song_no.tolist()

okt = Okt()
result = []
result_titles_list = []
result_keyword_list = []
result_song_no_list = []


for i, line in enumerate(lyrics):
    try:
        d = okt.pos(line, norm=True, stem=True)
        r = []
        for word in d:
            if not word[1] in ['Josa', 'Eomi', 'Punctuation']:  # 조사, 이모티콘, 문장부호를 제외시키고 담기
                r.append(word[0])
        rl = (" ".join(r)).strip()
        result.append(rl)
        result_titles_list.append(titles_list[i])
        result_keyword_list.append(keyword_list[i])
        result_song_no_list.append(song_no_list[i])
    except Exception as ee:
        print(ee)


# DataFrame에 사용할 데이터
data = {
    "genre": result_keyword_list,
    "song_no": result_song_no_list,
    "title": result_titles_list,
    "analyzed_lyric": result
}

# 데이터프레임 생성
df_analyzed = pd.DataFrame(data)

# 데이터프레임을 엑셀 파일로 내보내기
df_analyzed.to_excel("song_detail_analyzed.xlsx", index=False, engine="openpyxl", encoding='utf-8')

# with open('song_detail.txt', 'w', encoding='utf-8') as file:
#     for item in result:
#         file.write("%s\n" % item)