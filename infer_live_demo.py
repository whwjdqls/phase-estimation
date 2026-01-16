PROJECT_ID = "iitp-class-team-3-473114" 

# 가져올 데이터셋과 테이블명을 입력하세요.
DATASET_ID = "SBS_Data"
TABLE_ID = "live_demo"

# (옵션) 서비스 계정 키 JSON을 사용하는 경우에만 아래 주석 해제 및 경로 입력
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./path/to/your-service-account-key.json"

from google.cloud import bigquery
import pandas as pd
import torch
import pandas as pd
import numpy as np
from model.lstm import FlightPhaseLSTM
# 설정 정보
PROJECT_ID = "iitp-class-team-3-473114"
DATASET_ID = "SBS_Data"
TABLE_ID = "live_demo"

def get_bq_data(target_hex):
    """
    특정 HexIdent의 데이터를 BigQuery에서 가져와 1분 단위로 정제하여 반환합니다.
    """
    client = bigquery.Client(project=PROJECT_ID)

    # SQL 쿼리: 입력받은 target_hex에 대해서만 필터링 후 로직 수행
    query = f"""
        WITH Merged_Sec_Data AS (
            -- [1단계] 초 단위 병합 & 특정 HexIdent 필터링
            SELECT
                HexIdent,
                Date_MSG_Logged,
                Time_MSG_Logged,
                -- 정렬 및 그룹화를 위한 Datetime 생성
                DATETIME(Date_MSG_Logged, Time_MSG_Logged) AS Full_DateTime,
                
                -- 데이터 병합 (MAX)
                MAX(Altitude) AS Altitude,
                MAX(GroundSpeed) AS GroundSpeed,
                MAX(Track) AS Track,
                MAX(Latitude) AS Latitude,
                MAX(Longitude) AS Longitude,
                MAX(VerticalRate) AS VerticalRate,
                MAX(IsOnGround) AS IsOnGround
            FROM
                `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
            WHERE
                HexIdent = '{target_hex}'  -- 여기서 타겟 항공기만 골라냅니다 (속도/비용 최적화)
            GROUP BY
                HexIdent,
                Date_MSG_Logged,
                Time_MSG_Logged
        ),

        Ranked_Data AS (
            -- [2단계] 분 단위 리샘플링 (매분 00초에 가장 가까운 데이터 선택)
            SELECT
                *,
                ROW_NUMBER() OVER(
                    PARTITION BY DATETIME_TRUNC(Full_DateTime, MINUTE)
                    ORDER BY Full_DateTime ASC
                ) AS rn
            FROM
                Merged_Sec_Data
        )

        -- [3단계] 최종 선택 및 시간순 정렬
        SELECT
            HexIdent,
            Date_MSG_Logged,
            Time_MSG_Logged,
            Altitude,
            GroundSpeed,
            Track,
            Latitude,
            Longitude,
            VerticalRate,
            IsOnGround
        FROM
            Ranked_Data
        WHERE
            rn = 1
        ORDER BY
            Full_DateTime ASC;
    """

    print(f"[preprocess of flight {target_hex}]")

    try:
        query_job = client.query(query)
        df = query_job.to_dataframe()
        return df
    except Exception as e:
        print(f"에러 발생: {e}")
        return None

# --- 데이터 다운로드 ---
# 찾고 싶은 항공기의 Hex Code를 입력하세요
target_aircraft = "AA3579"  

df = get_bq_data(target_aircraft)

if df is not None:
    if df.empty:
        print("해당 HexIdent의 데이터가 없습니다.")
    else:
        print("Data loaded and preprocessed")
        print(f"Data size: {df.shape}")
        print("-" * 30)
        print(df.head())
        
        
        # 여기서 부터 정빈 코드 # 
        df = df.ffill().bfill()
        features = [
            "Altitude", 
            "GroundSpeed", 
            "Track", 
            "Latitude", 
            "Longitude", 
            "VerticalRate"
        ]

        data_np = df[features].astype(float).values 

        tensor_x = torch.tensor(data_np, dtype=torch.float32)

        n = tensor_x.shape[0]
        prefix_tensor = torch.full((n, 1), 60.0)

        final_tensor = torch.cat([prefix_tensor, tensor_x], dim=1)


        # print("Final Tensor Shape:", final_tensor.shape)
        # print(final_tensor[:2]) 

        # 모델 불러오기
        loaded = np.load('./stats.npz', allow_pickle=True)
        means = torch.tensor(loaded['mean'])
        stds = torch.tensor(loaded['std'])
        phase_map = loaded['phase_map'].item()
        idx_to_phase = {v: k for k, v in phase_map.items()}
        print("Phase Map:", phase_map)
        
        print("idx_to_phase:", idx_to_phase)
        
        # 모델 로드 (수정된 클래스 정의가 메모리에 있어야 함)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = FlightPhaseLSTM(7, 64, 2, len(phase_map))
        model.load_state_dict(torch.load('./best_flight_phase_lstm.pth'))
        model.to(device)
        model.eval()

        # 데이터 준비 (정규화 완료된 상태 가정)
        # final_tensor shape: (n, 7)
        data_normalized = (final_tensor.cpu() - means) / stds
        data_normalized_tensor = data_normalized.to(device)

        # 초기 기억 (Hidden State)은 없음
        hidden = None 

        print(f"Start Live Inference (Total steps: {len(data_normalized_tensor)})")
        print("-" * 50)

        with torch.no_grad():
            for t in range(len(data_normalized_tensor)):
                
                # (1) 현재 시점의 입력 하나 가져오기
                # (Batch=1, Seq=1, Feature=7) 형태로 변환
                input_step = data_normalized_tensor[t].view(1, 1, -1)
                
                # (2) 예측 수행 (이전 hidden을 넣고, 새 hidden을 받음)
                logits, hidden = model(input_step, hidden)
                
                # (3) 결과 해석
                _, predicted = torch.max(logits, 1)
                predicted_phase = idx_to_phase[predicted.item()]
                
                # (4) 출력 (현재 고도 등과 함께 출력하면 더 보기 좋습니다)
                curr_alt = final_tensor[t, 0].item() # Altitude (원본 값)
                print(f"[Step {t:03d}] Alt: {curr_alt:.0f}ft -> Phase: {predicted_phase}")