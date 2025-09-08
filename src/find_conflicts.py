# find_conflicts.py
import pandas as pd
import os
import math
from itertools import combinations

# --- 設定區 ---
CSV_PATH = os.path.join('..', 'output', 'tracking_results.csv')
OUTPUT_CSV_PATH = os.path.join('..', 'output', 'raw_iou_conflicts.csv')

VIDEO_FPS = 29.83
PREDICTION_HORIZON_SECONDS = 2.0
TIME_STEP = 0.1

# === 新增：方案2的設定 ===
# 用於移動平均的速度平滑化窗格大小
SMOOTHING_WINDOW_SIZE = 3 

# === 新增：方案3的設定 ===
# 速度向量點積的門檻值。大於此值代表兩物件大致朝同方向或垂直方向，非互相靠近
# 設為 0 表示只處理夾角大於90度的情況 (互相靠近)
DOT_PRODUCT_THRESHOLD = 0

# --- 核心函式 ---
def calculate_iou(box_a, box_b):
    """計算兩個邊界框的Intersection over Union (IoU)"""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    if inter_area == 0:
        return 0.0

    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    
    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou

def predict_ttc(obj1, obj2):
    """預測兩個物件的TTC，返回TTC秒數或None"""
    for t in range(1, int(PREDICTION_HORIZON_SECONDS / TIME_STEP) + 1):
        time_in_future = t * TIME_STEP
        
        # === 更新：使用平滑後的速度進行預測 ===
        pred_box1_x1 = obj1['x1'] + obj1['vx_smooth'] * VIDEO_FPS * time_in_future
        pred_box1_y1 = obj1['y1'] + obj1['vy_smooth'] * VIDEO_FPS * time_in_future
        pred_box1_x2 = obj1['x2'] + obj1['vx_smooth'] * VIDEO_FPS * time_in_future
        pred_box1_y2 = obj1['y2'] + obj1['vy_smooth'] * VIDEO_FPS * time_in_future
        
        pred_box2_x1 = obj2['x1'] + obj2['vx_smooth'] * VIDEO_FPS * time_in_future
        pred_box2_y1 = obj2['y1'] + obj2['vy_smooth'] * VIDEO_FPS * time_in_future
        pred_box2_x2 = obj2['x2'] + obj2['vx_smooth'] * VIDEO_FPS * time_in_future
        pred_box2_y2 = obj2['y2'] + obj2['vy_smooth'] * VIDEO_FPS * time_in_future
        
        pred_box1 = [pred_box1_x1, pred_box1_y1, pred_box1_x2, pred_box1_y2]
        pred_box2 = [pred_box2_x1, pred_box2_y1, pred_box2_x2, pred_box2_y2]
        
        if calculate_iou(pred_box1, pred_box2) > 0:
            return time_in_future
            
    return None

# --- 主程式 ---
try:
    df = pd.read_csv(CSV_PATH)
    print(f"成功讀取CSV數據檔案: {CSV_PATH}")
except FileNotFoundError:
    print(f"錯誤：找不到CSV檔案於路徑: {CSV_PATH}")
    exit()

# 預處理以計算速度
df = df.sort_values(by=['track_id', 'frame'])
df['center_x'] = (df['x1'] + df['x2']) / 2
df['center_y'] = (df['y1'] + df['y2']) / 2
df['prev_x'] = df.groupby('track_id')['center_x'].shift(1)
df['prev_y'] = df.groupby('track_id')['center_y'].shift(1)
df = df.dropna()
df['vx'] = df['center_x'] - df['prev_x']
df['vy'] = df['center_y'] - df['prev_y']

# === 新增：方案2 - 對vx, vy進行移動平均平滑 ===
df['vx_smooth'] = df.groupby('track_id')['vx'].transform(lambda x: x.rolling(SMOOTHING_WINDOW_SIZE, min_periods=1).mean())
df['vy_smooth'] = df.groupby('track_id')['vy'].transform(lambda x: x.rolling(SMOOTHING_WINDOW_SIZE, min_periods=1).mean())
print(f"已使用 {SMOOTHING_WINDOW_SIZE} 幀的窗格大小完成速度平滑化。")

raw_conflict_signals = []
grouped = df.groupby('frame')

print("開始執行帶有預過濾的原始碰撞風險偵測...")
for frame_id, frame_data in grouped:
    objects_in_frame = frame_data.to_dict('records')
    
    for obj1, obj2 in combinations(objects_in_frame, 2):
        
        # === 新增：方案3 - 行駛方向過濾 (向量點積) ===
        v1 = (obj1['vx_smooth'], obj1['vy_smooth'])
        v2 = (obj2['vx_smooth'], obj2['vy_smooth'])
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        
        # 如果點積大於門檻值，代表非對向行駛，跳過此組合
        if dot_product > DOT_PRODUCT_THRESHOLD:
            continue

        ttc = predict_ttc(obj1, obj2)
        
        if ttc is not None:
            id1, id2 = sorted((obj1['track_id'], obj2['track_id']))
            raw_conflict_signals.append({
                "frame_of_detection": frame_id,
                "track_id_1": id1,
                "class_1": obj1['class'] if obj1['track_id'] == id1 else obj2['class'],
                "track_id_2": id2,
                "class_2": obj2['class'] if obj2['track_id'] == id2 else obj1['class'],
                "predicted_TTC_seconds": ttc
            })

print(f"分析完成！總共偵測到 {len(raw_conflict_signals)} 筆原始碰撞訊號。")

if raw_conflict_signals:
    risk_df = pd.DataFrame(raw_conflict_signals)
    risk_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"成功！原始碰撞訊號已儲存至： {OUTPUT_CSV_PATH}")
else:
    print("在設定的條件下，未偵測到任何原始碰撞訊號。")