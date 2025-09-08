import cv2
import os

# --- 設定區 ---
# 使用相對路徑來定位影片檔案
# '..' 代表上一層目錄 (從 src/ 回到根目錄)
# os.path.join 會自動使用正確的斜線(/或\)，增加跨平台相容性
VIDEO_PATH = os.path.join('..', 'data', 'raw', '2021.12.17 新竹市建功人行天橋 朝東俯拍夜間車流 [OD07_FhpM6Q].mp4')

# --- 主程式 ---
# 讀取影片檔案
cap = cv2.VideoCapture(VIDEO_PATH)

# 檢查影片是否成功開啟
if not cap.isOpened():
    print(f"錯誤：無法開啟影片檔案。")
    print(f"請檢查檔案路徑是否正確: {VIDEO_PATH}")
else:
    print("成功開啟影片，按 'q' 鍵關閉視窗。")
    # 迴圈讀取影片的每一幀
    while True:
        ret, frame = cap.read()
        if not ret:
            print("影片播放完畢。")
            break

        # 在一個視窗中顯示當前畫面
        cv2.imshow("Video Playback Test - Press 'q' to quit", frame)

        # 等待25毫秒，如果使用者按下 'q' 鍵，就跳出迴圈
        # cv2.waitKey(25) 約等於每秒40幀(1000/25=40)，可以讓影片播放速度看起來比較正常
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# 釋放資源並關閉所有視窗
cap.release()
cv2.destroyAllWindows()