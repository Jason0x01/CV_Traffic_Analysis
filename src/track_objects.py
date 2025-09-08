import cv2
from ultralytics import YOLO
import os

# --- 設定區 ---
# 載入一個預訓練好的 YOLOv8 模型
model = YOLO('yolov8m.pt') 

# 定義影片檔案的路徑
VIDEO_PATH = os.path.join('..', 'data', 'raw', '2021.12.17 新竹市建功人行天橋 朝東俯拍夜間車流 [OD07_FhpM6Q].mp4')

# --- 主程式 ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"錯誤：無法開啟影片檔案於路徑: {VIDEO_PATH}")
else:
    print("成功開啟影片，開始進行物件追蹤...")
    print("按 'q' 鍵關閉視窗。")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # *** AI 追蹤核心 ***
            # 將 model() 改為 model.track() 來啟用追蹤功能
            # persist=True 告訴追蹤器，在連續的畫面之間要記住物件的ID
            results = model.track(frame, persist=True)

            # 將追蹤結果（包括標註框、追蹤ID等）畫在原始畫面上
            annotated_frame = results[0].plot()

            # 顯示已經被標註過的畫面
            cv2.imshow("YOLOv8 Tracking - Press 'q' to quit", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            print("影片已結束。")
            break

cap.release()
cv2.destroyAllWindows()