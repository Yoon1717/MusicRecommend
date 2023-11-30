from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from bs4 import BeautifulSoup
import pandas as pd

# 멜론 월간차트 크롤링 data-song-no 노래 고유번호 저장

# 1.장르별 음악 코드가져오기
# 2.코드로 가사가져오기
# Melon 곡 정보 페이지 URL
url = 'https://www.melon.com/chart/month/index.htm?classCd='
song_list = []

def get_genre(query):
    driver = webdriver.Chrome("../chromedriver.exe")
    driver.implicitly_wait(3)
    driver.get(url + query)
    table = driver.find_element(By.TAG_NAME,'table')
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    time.sleep(1)

    # 데이터 수집 및 리스트에 추가
    song_elements = soup.find_all('tr', attrs={'data-song-no' : True})
    for song in song_elements:
        song_no = song['data-song-no']
        print(f'data-song-no: {song_no}')
        song_list.append({'keyword': query, 'song-no': song_no})



keyword = ['GN0100', 'GN0200', 'GN0300', 'GN0400',
           'GN0500', 'GN0600', 'GN0700', 'GN0800']
# 발라드, 댄스, 랩/힙합, R&B/Soul, 인디음악, 록/메탈, 트로트, 포크/블루스
for k in keyword:
    get_genre(k)
   
#  데이터를 데이터프레임으로변환, 데이터프레임 엑셀파일로 저장
df = pd.DataFrame(song_list)
df.to_excel('melon_month_craw.xlsx', index=False)

# 'GN0100' : 발라드
# 'GN0200' : 댄스
# 'GN0300' : 랩/힙합
# 'GN0400' : R&B/Soul
# 'GN0500' : 인디음악
# 'GN0600' : 록/메탈
# 'GN0700' : 트로트
# 'GN0800' : 포크/블루스
