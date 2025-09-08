# summarize_events.py
import pandas as pd
import os

# --- 設定區 ---
INPUT_CSV_PATH = os.path.join('..', 'output', 'raw_iou_conflicts.csv')
OUTPUT_SUMMARY_PATH = os.path.join('..', 'output', 'unique_conflict_summary.csv')
# 如果同一個衝突事件中斷了超過 N 幀，我們就視為一個新的事件
FRAME_GAP_THRESHOLD = 60 # (約2秒) - 已根據建議加大此值，您可以再調整
# === 新增：方案1的設定 ===
# 獨立事件的最小持續時間門檻（幀），過於短暫的事件將被過濾
MINIMUM_DURATION_THRESHOLD = 10 # (約0.33秒)

# --- 主程式 ---
try:
    df = pd.read_csv(INPUT_CSV_PATH)
    print(f"成功讀取原始碰撞訊號檔案: {INPUT_CSV_PATH}")
except FileNotFoundError:
    print(f"錯誤：找不到檔案於路徑: {INPUT_CSV_PATH}")
    exit()

df['pair'] = df.apply(lambda row: tuple(sorted((row['track_id_1'], row['track_id_2']))), axis=1)

unique_events = []
for pair, group in df.groupby('pair'):
    group = group.sort_values(by='frame_of_detection')
    if len(group) == 0:
        continue
    
    start_frame = group.iloc[0]['frame_of_detection']
    min_ttc = group.iloc[0]['predicted_TTC_seconds']
    
    for i in range(1, len(group)):
        prev_frame = group.iloc[i-1]['frame_of_detection']
        curr_frame = group.iloc[i]['frame_of_detection']
        
        if curr_frame - prev_frame > FRAME_GAP_THRESHOLD:
            unique_events.append({
                "object_pair": pair,
                "start_frame": start_frame,
                "end_frame": prev_frame,
                "duration_frames": prev_frame - start_frame + 1,
                "min_TTC_seconds": min_ttc
            })
            start_frame = curr_frame
            min_ttc = group.iloc[i]['predicted_TTC_seconds']
        else:
            min_ttc = min(min_ttc, group.iloc[i]['predicted_TTC_seconds'])
    
    unique_events.append({
        "object_pair": pair,
        "start_frame": start_frame,
        "end_frame": group.iloc[-1]['frame_of_detection'],
        "duration_frames": group.iloc[-1]['frame_of_detection'] - start_frame + 1,
        "min_TTC_seconds": min_ttc
    })

print(f"初步彙總完成！共得到 {len(unique_events)} 次潛在的獨立事件。")

if unique_events:
    summary_df = pd.DataFrame(unique_events)
    summary_df = summary_df.drop_duplicates()
    
    # === 新增：方案1 - 只保留持續時間足夠長的事件 ===
    original_count = len(summary_df)
    summary_df = summary_df[summary_df['duration_frames'] >= MINIMUM_DURATION_THRESHOLD]
    filtered_count = len(summary_df)
    print(f"透過持續時間過濾 (>{MINIMUM_DURATION_THRESHOLD}幀)，將事件從 {original_count} 次精煉為 {filtered_count} 次。")

    summary_df.to_csv(OUTPUT_SUMMARY_PATH, index=False)
    print(f"成功！最終的獨立事件摘要已儲存至： {OUTPUT_SUMMARY_PATH}")