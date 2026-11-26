import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="自分専用 競馬AI", layout="wide")
st.title("🏇 自分専用 競馬AI予想")

# 学習済みAIモデルの読み込み
@st.cache_resource
def load_ai_model():
    return joblib.load("keiba_ai_model.pkl")

model = load_ai_model()

# レース選択・タイトル設定
col_r1, col_r2 = st.columns(2)
with col_r1:
    kaisai = st.selectbox("開催場", ("東京", "京都", "中山", "阪神", "中京", "小倉", "新潟", "福島", "函館", "札幌"))
with col_r2:
    race_name = st.text_input("レース名", value="11R ジャパンカップ(G1)")

st.subheader(f"📍 {kaisai} {race_name}")

# 初期出馬表データ（丸カッコ使用で安全定義）
if "horse_df" not in st.session_state:
    st.session_state.horse_df = pd.DataFrame({
        "枠番": (1, 2, 3, 4, 5, 6, 7, 8),
        "馬番": (1, 2, 3, 4, 5, 6, 7, 8),
        "馬名": ("サンプルエース", "スピードスター", "穴馬ハンター", "ディープインパクトⅡ", "ステイヤーキング", "ダッシュボーイ", "ミラクルガール", "ダークホース"),
        "騎手": ("ルメール", "川田将雅", "武豊", "戸崎圭太", "横山武史", "松山弘平", "坂井瑠星", "菅原明良"),
        "単勝オッズ": (2.4, 4.8, 28.5, 6.2, 12.0, 15.4, 9.8, 45.0),
        "斤量": (58.0, 58.0, 57.0, 56.0, 58.0, 57.0, 55.0, 57.0),
        "前走着順": (1, 2, 6, 1, 4, 8, 3, 5),
        "前走上がり3F順位": (1, 3, 2, 2, 5, 9, 4, 1),
        "同コース複勝率": (0.50, 0.40, 0.35, 0.45, 0.25, 0.20, 0.30, 0.33),
        "騎手勝率": (0.22, 0.20, 0.14, 0.15, 0.14, 0.13, 0.14, 0.11)
    })

# AIによるスコア予測の実行
features = ("枠番", "馬番", "単勝オッズ", "斤量", "前走着順", "前走上がり3F順位", "同コース複勝率", "騎手勝率")
input_df = st.session_state.horse_df.copy()

# 3着以内確率をAIモデルで予測
X_pred = input_df.loc[:, features]
probs = model.predict_proba(X_pred)[:, 1]
input_df = input_df.assign(
    的中スコア=np.round(probs * 100, 1),
    回収スコア=np.round((probs * (input_df.get("単勝オッズ") ** 0.55)) * 100, 1)
)

# スコアモード切替
mode = st.radio(
    "AIスコア表示モード",
    ("🎯 的中重視モード（3着内確率）", "💰 回収率重視モード（期待値）"),
    horizontal=True
)

if "的中" in mode:
    input_df = input_df.assign(表示スコア=input_df.get("的中スコア"))
else:
    input_df = input_df.assign(表示スコア=input_df.get("回収スコア"))

# スコア順に並び替えて印を付与
ranked_df = input_df.sort_values("表示スコア", ascending=False).reset_index(drop=True)
marks = ("◎", "○", "▲", "△", "☆")
mark_list = tuple(marks[i] if i < len(marks) else "" for i in range(len(ranked_df)))
ranked_df = ranked_df.assign(印=mark_list)

# netkeiba風タブ構成
tab1, tab2, tab3 = st.tabs(("📋 AI予想・出馬表", "🐎 馬別詳細・根拠", "✏️ 出走馬の編集・追加"))

with tab1:
    st.caption("※「✏️ 出走馬の編集・追加」タブでオッズや前走成績を変えると、AIスコアも自動更新されます。")
    show_cols = ("印", "枠番", "馬番", "馬名", "騎手", "単勝オッズ", "的中スコア", "回収スコア")
    st.dataframe(ranked_df.loc[:, show_cols], use_container_width=True, hide_index=True)

with tab2:
    selected_horse = st.selectbox("詳細を見る馬を選択", ranked_df.get("馬名"))
    info = ranked_df.query("馬名 == @selected_horse").squeeze()
    
    st.subheader(f"{info.get('印')} {int(info.get('馬番'))}番 {info.get('馬名')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("単勝オッズ", f"{info.get('単勝オッズ')}倍")
    c2.metric("🎯 的中スコア", f"{info.get('的中スコア')}点")
    c3.metric("💰 回収スコア", f"{info.get('回収スコア')}pt")
    
    st.write("---")
    st.write("**AIが評価した詳細データ：**")
    st.write(f"・**騎手 / 斤量：** {info.get('騎手')}（勝率 {int(info.get('騎手勝率')*100)}%） / {info.get('斤量')}kg")
    st.write(f"・**前走成績：** {int(info.get('前走着順'))}着（上がり3F順位：{int(info.get('前走上がり3F順位'))}位）")
    st.write(f"・**同コース複勝率：** {int(info.get('同コース複勝率')*100)}%")

with tab3:
    st.write("表のセルをタップすると数値を直接書き換えられます。一番下の「+」で行を追加できます。")
    edited_df = st.data_editor(
        st.session_state.horse_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True
    )
    if st.button("🔄 この内容でAI再計算する", type="primary"):
        st.session_state.horse_df = edited_df
        st.rerun()
