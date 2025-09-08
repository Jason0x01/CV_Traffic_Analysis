import cv2
from ultralytics import YOLO
import os
import pandas as pd

# --- 設定區 ---
model = YOLO('yolov8m.pt') 
VIDEO_PATH = os.path.join('..', 'data', 'raw', '2021.12.17 新竹市建功人行天橋 朝東俯拍夜間車流 [OD07_FhpM6Q].mp4')

# --- 主程式 ---
cap = cv2.VideoCapture(VIDEO_PATH)

# 用來儲存所有偵測結果的列表
tracking_data = []
frame_count = 0

if not cap.isOpened():
    print(f"錯誤：無法開啟影片檔案於路徑: {VIDEO_PATH}")
else:
    print("成功開啟影片，開始提取追蹤數據...")
    print("按 'q' 鍵可以提早結束。")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_count += 1 # 幀數計數器
            
            # 進行物件追蹤
            results = model.track(frame, persist=True)
            
            # *** 數據提取核心 ***
            # 檢查是否有追蹤ID
            if results[0].boxes.id is not None:
                # 獲取標註框、追蹤ID和類別
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                
                # 遍歷每一個偵測到的物件
                for box, track_id, cls in zip(boxes, track_ids, clss):
                    # 將這一個物件的資訊，加入到我們的列表中
                    tracking_data.append({
                        "frame": frame_count,
                        "track_id": track_id,
                        "class": model.names[cls], # 將類別ID轉換為名稱(例如: 'car')
                        "x1": box[0].item(),
                        "y1": box[1].item(),
                        "x2": box[2].item(),
                        "y2": box[3].item()
                    })
            
            # (視覺化部分，可選) 將結果畫出來讓我們能即時看到
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Tracking - Extracting Data", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("影片已結束。")
            break

cap.release()
cv2.destroyAllWindows()

# --- 數據儲存 ---
print(f"數據提取完成，總共處理了 {frame_count} 幀。")
print("正在將結果儲存為CSV檔案...")

# 將列表轉換為 Pandas DataFrame
df = pd.DataFrame(tracking_data)

# 定義輸出的CSV檔案路徑
OUTPUT_CSV_PATH = os.path.join('..', 'output', 'tracking_results.csv')

# 將DataFrame儲存為CSV檔案
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"成功！數據已儲存至： {OUTPUT_CSV_PATH}")