import cv2
import os

# --- 設定區 ---
# 請確認路徑與您的檔案位置一致
VIDEO_PATH = os.path.join('..', 'data', 'raw', '2021.12.17 新竹市建功人行天橋 朝東俯拍夜間車流 [OD07_FhpM6Q].mp4')

# --- 主程式 ---
# 讀取影片
cap = cv2.VideoCapture(VIDEO_PATH)

# 檢查是否成功開啟
if not cap.isOpened():
    print(f"錯誤：找不到影片檔案於 {VIDEO_PATH}")
else:
    # 使用 cv2.CAP_PROP_FPS 來獲取影片的幀率屬性
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"影片 '{os.path.basename(VIDEO_PATH)}' 的真實幀率是： {fps:.2f} FPS")

# 釋放資源
cap.release()