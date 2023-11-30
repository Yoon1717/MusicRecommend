from selenium import webdriver
import time
from bs4 import BeautifulSoup
import pandas as pd
# from selenium.webdriver.common.by import By


# 옵션 생성
options = webdriver.ChromeOptions()
# 창 숨기는 옵션 추가
options.add_argument("headless")

url = 'https://www.melon.com/song/detail.htm?songId='

df = pd.read_excel("melon_month_craw.xlsx", engine='openpyxl')
song_no = df['song-no'].values  # 곡 고유번호
keyword = df['keyword'].values  # 장르 번호
url_list = []
for i, v in enumerate(song_no):
    url_list.append([url + str(v), str(v), keyword[i]])

song_detail_list = []
def get_song_detail(url, code, key):
    driver = webdriver.Chrome("../chromedriver.exe", options=options)
    driver.implicitly_wait(3)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    time.sleep(1)

    # 가사 정보를 가져오기 위해 해당 div를 찾음
    lyric_div = soup.find('div', {'id': 'd_video_summary'})

    lyric = ""
    for con in lyric_div.contents:              # html을 가져옴
        if 'NavigableString' in str(type(con)): # NavigableString : BeautifulSoup의 텍스트구조
            if lyric != "":
                lyric += " "
            lyric += con.text.strip()
    print(lyric)

    # 제목 정보를 가져오기 위해 해당 div를 찾음
    title_div = soup.find('div', {'class': 'song_name'})
    title = title_div.find('strong').find_next_sibling(string=True).strip()
    print(title)

    song_detail_list.append({'keyword': key,
                                    'song_no': code,
                                    'title': title,
                                    'lyric': lyric})
    driver.close()



for url in url_list:
    try:
        get_song_detail(url[0], url[1], url[2])
    except Exception as e:
        print(str(e))



df = pd.DataFrame(song_detail_list)
df.to_excel('song_detail.xlsx', index=False)



