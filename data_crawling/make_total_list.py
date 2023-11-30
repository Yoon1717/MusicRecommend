import pandas as pd

# 두 개의 CSV 파일 로드 (가능한 인코딩: 'cp949')
df1 = pd.read_csv('analyzed.csv', encoding='cp949')  # 1번 파일
df2 = pd.read_csv('origin_detail.csv', encoding='cp949')  # 2번 파일

# song_no를 기준으로 두 데이터프레임을 조인 (merge)
result = df1.merge(df2[['song_no', 'origin_lyric']], on='song_no', how='left')

# 'origin_lyric' 열이 비어 있는 경우 '가사를 찾을 수 없습니다'로 채우기
result['origin_lyric'] = result['origin_lyric'].fillna('가사를 찾을 수 없습니다')

# 결과를 새로운 CSV 파일로 저장 ('utf-8'로 저장 가능)
result.to_csv('total_list.csv', index=False, encoding='utf-8')

print(result)
