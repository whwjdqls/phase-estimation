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
    'IsOnGround'
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