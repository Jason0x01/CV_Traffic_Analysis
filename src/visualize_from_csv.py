import cv2
import pandas as pd
import os

# --- 設定區 ---
# 影片路徑，需要跟生成數據時的影片完全一樣
VIDEO_PATH = os.path.join('..', 'data', 'raw', '2021.12.17 新竹市建功人行天橋 朝東俯拍夜間車流 [OD07_FhpM6Q].mp4')
# 我們之前儲存的CSV數據路徑
CSV_PATH = os.path.join('..', 'output', 'tracking_results.csv')

# --- 主程式 ---
# 讀取CSV檔案到Pandas DataFrame
try:
    df = pd.read_csv(CSV_PATH)
    print("成功讀取CSV數據檔案。")
except FileNotFoundError:
    print(f"錯誤：找不到CSV檔案於路徑: {CSV_PATH}")
    print("請先執行 analyze_data.py 來生成數據。")
    exit() # 如果找不到檔案，就結束程式

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

if not cap.isOpened():
    print(f"錯誤：無法開啟影片檔案於路徑: {VIDEO_PATH}")
else:
    print("成功開啟影片，開始根據CSV數據進行視覺化...")
    print("按 'q' 鍵關閉視窗。")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            
            # *** 數據驅動視覺化核心 ***
            # 從DataFrame中，篩選出「當前這一幀」的所有偵測數據
            detections_in_frame = df[df['frame'] == frame_count]
            
            # 如果這一幀有數據，就把標註框和ID畫上去
            if not detections_in_frame.empty:
                # 遍歷篩選出的每一筆數據
                for index, row in detections_in_frame.iterrows():
                    # 從row中讀取座標和ID
                    x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
                    track_id = int(row['track_id'])
                    class_name = row['class']
                    
                    # 在畫面上畫出標註框 (綠色)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 在畫面上標示出ID和類別
                    label = f"id:{track_id} {class_name}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 顯示已經被我們手動標註過的畫面
            cv2.imshow("Visualization from CSV", frame)

            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            print("影片已結束。")
            break

cap.release()
cv2.destroyAllWindows()