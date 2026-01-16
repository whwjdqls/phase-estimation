import pandas as pd
import os

# 1. 파일 경로 설정
file_path = '/scratch2/tjgus0408/CMU/data/Labeled_OpenAP.csv'

# 2. 사용할 컬럼 목록 정의
selected_columns = [
    'HexIdent', 
    'Date_MSG_Generated', 
    'Time_MSG_Generated',
    'Date_MSG_Logged', 
    'Time_MSG_Logged', 
    'Altitude', 
    'GroundSpeed',
    'Track', 
    'Latitude', 
    'Longitude', 
    'VerticalRate', 
    'IsOnGround',
    "Phase"
]

# 3. 데이터 로드 (파일이 존재하는지 확인 후 로드)
if os.path.exists(file_path):
    try:
        # usecols 파라미터를 사용하여 필요한 컬럼만 읽어오면 메모리를 절약할 수 있습니다.
        df = pd.read_csv(file_path, usecols=selected_columns)
        
        print("✅ 데이터 로드 성공!")
        print(f"데이터 크기 (행, 열): {df.shape}")
        
        # 데이터 미리보기
        print("\n[상위 5개 행]")
        print(df.head())
        
        # 데이터 타입 및 결측치 확인
        print("\n[데이터 정보]")
        print(df.info())
        
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
else:
    print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
    
    
    
    
# 1. 날짜와 시간을 합쳐서 하나의 datetime 컬럼 생성 (Date_MSG_Generated + Time_MSG_Generated 사용)
# 데이터가 문자열(object) 형태여도 pd.to_datetime이 똑똑하게 변환해줍니다.
print("⏳ 날짜와 시간 병합 중...")
df['Timestamp'] = pd.to_datetime(df['Date_MSG_Generated'] + ' ' + df['Time_MSG_Generated'])

df_multi = df.sort_values(by=['HexIdent', 'Timestamp']).set_index(['HexIdent', 'Timestamp'])

# 1. 고유한 값 추출 (NumPy 배열 형태)
# df_multi = df.sort_values(by=['HexIdent', 'Timestamp']).set_index(['HexIdent', 'Timestamp'])
unique_hex_ids = df["HexIdent"].unique()

# 2. (선택 사항) 파이썬 리스트 형태로 변환 (다루기 더 편함)
hex_id_list = unique_hex_ids.tolist()

# 결과 확인


import pandas as pd

# ---------------------------------------------------------
# [가정] df가 이미 로드되어 있고, Timestamp 컬럼이 datetime 형식이라고 가정
# (만약 MultiIndex라면 reset_index()를 먼저 수행하여 컬럼으로 내려줍니다)
if 'HexIdent' not in df.columns or 'Timestamp' not in df.columns:
    df = df.reset_index()
# ---------------------------------------------------------

# 1. 데이터 길이가 20개 이하인 HexIdent 제거 (Filtering)
# ---------------------------------------------------------
print(f"전처리 전 데이터 크기: {len(df)}")

# 각 HexIdent별 데이터 개수 계산
counts = df.groupby('HexIdent').size()

# 20개보다 큰 HexIdent만 추출 (Index 리스트)
valid_hex_ids = counts[counts > 20].index

# 해당 HexIdent를 가진 행만 남김
df_filtered = df[df['HexIdent'].isin(valid_hex_ids)].copy()

print(f"20개 이하 제거 후 데이터 크기: {len(df_filtered)}")


# 2. 1시간 이상 공백이 있으면 분리 (Segmentation)
# ---------------------------------------------------------

# (중요) HexIdent별, 그리고 시간순으로 정렬이 되어 있어야 정확히 계산됨
df_filtered = df_filtered.sort_values(by=['HexIdent', 'Timestamp'])

# (1) 이전 행과의 시간 차이 계산 (HexIdent 그룹별로 수행)
# HexIdent가 바뀌는 지점은 diff가 이상하게 나올 수 있으나, 어차피 그룹별로 처리하므로 상관없음
df_filtered['time_diff'] = df_filtered.groupby('HexIdent')['Timestamp'].diff()

# (2) 시간 차이가 1시간('1h')보다 큰 경우를 찾음 (True/False)
# 첫 번째 행(NaT)은 새로운 시작이므로 False(0) 처리 후 로직 적용
threshold = pd.Timedelta(minutes=30)
df_filtered['is_new_segment'] = (df_filtered['time_diff'] > threshold).fillna(False)

# (3) 누적 합(cumsum)을 통해 그룹 내에서 Segment 번호 부여
# 예: [False, False, True, False] -> [0, 0, 1, 1]
df_filtered['segment_id'] = df_filtered.groupby('HexIdent')['is_new_segment'].cumsum()


# 3. 최종 Instance ID 생성 (HexIdent + Segment 번호)
# ---------------------------------------------------------
# 예: A4CE9D_0, A4CE9D_1 ...
df_filtered['Unique_ID'] = df_filtered['HexIdent'] + "_" + df_filtered['segment_id'].astype(str)

# 불필요한 임시 컬럼 삭제
df_final = df_filtered.drop(columns=['time_diff', 'is_new_segment', 'segment_id'])

# 결과 확인
print(f"총 생성된 Instance(비행 단위) 개수: {df_final['Unique_ID'].nunique()}")
print(df_final[['HexIdent', 'Timestamp', 'Unique_ID']].head(10))

# ---------------------------------------------------------
# 4. 저장 및 활용 (Dictionary 형태로 변환 추천)
# ---------------------------------------------------------

# 각 Unique_ID를 Key로, 해당 데이터프레임을 Value로 갖는 딕셔너리 생성
instance_dict = {k: v for k, v in df_final.groupby('Unique_ID')}

# (선택 사항) 분할 후, 길이가 너무 짧은 Instance는 다시 제거하기
final_counts = df_final.groupby('Unique_ID').size()
valid_instances = final_counts[final_counts > 20].index
df_final_clean = df_final[df_final['Unique_ID'].isin(valid_instances)]

instance_dict = {k: v for k, v in df_final_clean.groupby('Unique_ID')}



total_nans = sum(df['Phase'].isna().sum() for df in instance_dict.values())
print(f"남은 Phase 결측치 개수: {total_nans}")