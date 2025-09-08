import streamlit as st
import pandas as pd
import os
import cv2
import ast
import matplotlib.pyplot as plt

# --- 字體設定 (解決中文亂碼問題) ---
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# --- 頁面基礎設定 ---
st.set_page_config(page_title="智慧交通風險分析儀表板", layout="wide")
st.title("智慧交通影像分析與風險評估系統")
st.write("這是一個基於電腦視覺與AI演算法的交通影像分析專案原型。")
st.write("---")

# --- 數據載入 (使用相對於本腳本的絕對路徑，確保雲端部署成功) ---
# 取得此腳本檔案 (app.py) 所在的目錄 (也就是 src/ 目錄)
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# 取得專案根目錄 (SCRIPT_DIR 的上一層)
PROJECT_ROOT = os.path.realpath(os.path.join(SCRIPT_DIR, '..'))

# 從專案根目錄建立指向資料檔的絕對路徑
TRACKING_CSV_PATH = os.path.join(PROJECT_ROOT, 'output', 'tracking_results.csv')
CONFLICT_CSV_PATH = os.path.join(PROJECT_ROOT, 'output', 'unique_conflict_summary.csv')
VIDEO_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', '2021.12.17 新竹市建功人行天橋 朝東俯拍夜間車流 [OD07_FhpM6Q].mp4')

@st.cache_data
def load_data(tracking_path, conflict_path):
    try:
        tracking_df = pd.read_csv(tracking_path)
        conflict_df = pd.read_csv(conflict_path)
        return tracking_df, conflict_df
    except FileNotFoundError:
        return None, None

tracking_df, conflict_df = load_data(TRACKING_CSV_PATH, CONFLICT_CSV_PATH)

if tracking_df is None or conflict_df is None:
    st.error("錯誤：找不到分析數據檔案。請先執行 src/ 資料夾中的所有分析程式。")
else:
    # --- 儀表板主體 ---
    st.header("專案核心數據總覽")

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_length_seconds = total_frames / fps
    cap.release()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("影片總長度 (秒)", f"{video_length_seconds:.2f}")
    col2.metric("總分析幀數", f"{total_frames}")
    col3.metric("偵測到的獨立物件總數", f"{tracking_df['track_id'].nunique()}")
    col4.metric("彙總出的獨立風險事件", f"{len(conflict_df)}")
    
    st.write("---")
    st.header("風險事件深度分析 (Deep Dive into Risk Events)")
    
    # ... (後續程式碼與之前版本相同，此處省略以保持簡潔) ...
    # (請確保您使用的是包含Matplotlib繪圖與所有修正的最新版本)
    st.subheader("風險事件類別分佈")
    conflict_df['pair_tuple'] = conflict_df['object_pair'].apply(ast.literal_eval)
    
    id_to_class_map = tracking_df.drop_duplicates(subset=['track_id']).set_index('track_id')['class'].to_dict()
    
    def get_class_pair(row):
        class1 = id_to_class_map.get(row['pair_tuple'][0])
        class2 = id_to_class_map.get(row['pair_tuple'][1])
        if class1 and class2:
            IGNORE_CLASSES = ['traffic light', 'train']
            if class1 in IGNORE_CLASSES or class2 in IGNORE_CLASSES:
                return None
            return str(tuple(sorted((class1, class2))))
        return None

    conflict_df['class_pair_str'] = conflict_df.apply(get_class_pair, axis=1)
    pair_counts = conflict_df['class_pair_str'].value_counts().dropna()
    
    fig, ax = plt.subplots(figsize=(12, 7))
    pair_counts.plot(kind='bar', ax=ax)
    ax.set_title("風險事件類別分佈", fontsize=18)
    ax.set_xlabel("物件組合", fontsize=14)
    ax.set_ylabel("事件次數", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    st.write("從上圖可見，在此場景中，不同用路人類別之間的潛在衝突分佈。")

    st.subheader("碰撞時間 (TTC) 分佈")
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    conflict_df['min_TTC_seconds'].hist(bins=20, ax=ax2)
    ax2.set_title("碰撞時間 (TTC) 分佈", fontsize=18)
    ax2.set_xlabel("預測碰撞時間 (秒)", fontsize=14)
    ax2.set_ylabel("事件次數", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(fig2)
    st.write("TTC分佈圖顯示了風險事件的緊急程度。")
    
    st.write("---")
    st.header("關鍵事件視覺化探索 (Key Event Visualization)")
    
    conflict_df['event_description'] = (
        "事件 ID: " + conflict_df.index.astype(str) + 
        " | 發生幀: " + conflict_df['start_frame'].astype(str) +
        " | 物件: " + conflict_df['object_pair'].astype(str) +
        " | 最短TTC: " + conflict_df['min_TTC_seconds'].round(2).astype(str) + "秒"
    )

    selected_event_desc = st.selectbox("請從下拉選單中選擇一個高風險事件來進行視覺化：", options=conflict_df['event_description'])

    if selected_event_desc:
        event_index = int(selected_event_desc.split(' ')[2])
        selected_event = conflict_df.iloc[event_index]
        
        frame_id = selected_event['start_frame']
        pair = selected_event['pair_tuple']
        id1, id2 = pair[0], pair[1]
        
        time_in_seconds = frame_id / fps
        
        st.info(f"您選擇了事件 **#{event_index}**。此事件發生在影片約 **{time_in_seconds:.2f}** 秒處，涉及物件 **ID {id1}** 與 **ID {id2}**。")

        cap = cv2.VideoCapture(VIDEO_PATH)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = cap.read()
        cap.release()

        if ret:
            event_objects = tracking_df[(tracking_df['frame'] == frame_id) & (tracking_df['track_id'].isin([id1, id2]))]
            
            for _, obj in event_objects.iterrows():
                x1, y1, x2, y2 = int(obj['x1']), int(obj['y1']), int(obj['x2']), int(obj['y2'])
                label = f"id:{obj['track_id']} {obj['class']}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3) 
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption=f"高風險事件發生瞬間 (第 {frame_id} 幀)", use_container_width=True)
        else:
            st.error("無法讀取指定幀的影片畫面。")
