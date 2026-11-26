import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="自分専用 競馬AI", layout="wide")
st.title("🏇 自分専用 競馬AI予想")

@st.cache_resource
def load_ai_model():
    return joblib.load("keiba_ai_model.pkl")

model = load_ai_model()

if os.path.exists("jra_races.csv"):
    all_df = pd.read_csv("jra_races.csv")
else:
    st.error("jra_races.csv が見つかりません。GitHubにアップロードしてください。")
    st.stop()

# レース番号列がない古いCSVでも自動補完して順番通りに並べる
if "レース番号" not in all_df.columns:
    all_df = all_df.assign(
        レース番号=all_df.get("レース").astype(str).str.extract(r"(\d+)").fillna(99).astype(int)
    )

# 1. 開催場とレースの選択（1R〜12R順）
col_r1, col_r2 = st.columns(2)
with col_r1:
    kaisai_list = tuple(all_df.get("開催場").unique())
    selected_kaisai = st.selectbox("開催場を選択", kaisai_list)
with col_r2:
    sub_df = all_df.query("開催場 == @selected_kaisai").sort_values("レース番号")
    race_list = tuple(sub_df.get("レース").unique())
    selected_race = st.selectbox("レースを選択", race_list)

st.subheader(f"📍 {selected_kaisai} {selected_race}")

race_df = sub_df.query("レース == @selected_race").sort_values("馬番").copy()

# 騎手勝率のマッピング
top_jockeys = {
    "ルメ": 0.24, "Lemaire": 0.24,
    "川田": 0.23, "Kawada": 0.23,
    "モレイラ": 0.25, "Moreira": 0.25,
    "戸崎": 0.16, "Tosaki": 0.16,
    "横山": 0.15, "Yokoyama": 0.15,
    "坂井": 0.16, "Sakai": 0.16,
    "武豊": 0.15, "Take": 0.15,
    "デムー": 0.18, "Demuro": 0.18,
    "松山": 0.15, "Matsuyama": 0.15,
    "鮫島": 0.13, "Sameshima": 0.13
}

def get_jockey_rate(name):
    for k, v in top_jockeys.items():
        if k in str(name):
            return v
    return 0.10

j_rates = tuple(get_jockey_rate(j) for j in race_df.get("騎手"))
np.random.seed(len(race_df) + int(race_df.get("馬番").sum()))
n = len(race_df)
est_odds = np.round(np.clip(2.0 / np.array(j_rates) + np.random.uniform(-1.5, 7.5, n), 1.5, 99.9), 1)

race_df = race_df.assign(
    単勝オッズ=est_odds,
    前走着順=np.random.randint(1, 9, n),
    前走上がり3F順位=np.random.randint(1, 9, n),
    同コース複勝率=np.round(np.array(j_rates) * 1.8, 2),
    騎手勝率=j_rates
)

state_key = f"df_{selected_kaisai}_{selected_race}"
if state_key not in st.session_state:
    st.session_state[state_key] = race_df

curr_df = st.session_state.get(state_key).copy()

# AIスコア計算
features = ("枠番", "馬番", "単勝オッズ", "斤量", "前走着順", "前走上がり3F順位", "同コース複勝率", "騎手勝率")
X_pred = curr_df.loc[:, features]
probs = model.predict_proba(X_pred)[:, 1]

curr_df = curr_df.assign(
    的中スコア=np.round(probs * 100, 1),
    回収スコア=np.round((probs * (curr_df.get("単勝オッズ") ** 0.55)) * 100, 1)
)

# 2. スコアモード切替
mode = st.radio(
    "AIスコア表示モード",
    ("🎯 的中重視モード（3着内確率）", "💰 回収率重視モード（期待値）"),
    horizontal=True
)
score_col = "的中スコア" if "的中" in mode else "回収スコア"

ranked_df = curr_df.sort_values(score_col, ascending=False).reset_index(drop=True)
marks = ("◎", "○", "▲", "△", "☆")
mark_list = tuple(marks[i] if i < len(marks) else "" for i in range(len(ranked_df)))
ranked_df = ranked_df.assign(印=mark_list)

# 3. タブ切り替え表示
tab1, tab2, tab3, tab4 = st.tabs(("📋 AI予想・出馬表", "🎯 おすすめ買い目", "🐎 馬別詳細", "✏️ オッズ・前走調整"))

with tab1:
    show_cols = ("印", "枠番", "馬番", "馬名", "騎手", "斤量", "単勝オッズ", "的中スコア", "回収スコア")
    st.dataframe(ranked_df.loc[:, show_cols], use_container_width=True, hide_index=True)

with tab2:
    st.subheader("🤖 AI自動提案フォーメーション")
    records = ranked_df.to_dict("records")
    if len(records) >= 4:
        honmei = records.pop(0)
        taikou = records.pop(0)
        tanana = records.pop(0)
        renka = records.pop(0)
        st.success(f"**軸馬（◎）：{int(honmei.get('馬番'))}番 {honmei.get('馬名')}**（{score_col}: {honmei.get(score_col)}）")
        st.write(f"**相手（○▲△）：** {int(taikou.get('馬番'))}番 {taikou.get('馬名')} / {int(tanana.get('馬番'))}番 {tanana.get('馬名')} / {int(renka.get('馬番'))}番 {renka.get('馬名')}")
        st.info(f"**推奨馬連（3点）：** {int(honmei.get('馬番'))} － {int(taikou.get('馬番'))}, {int(tanana.get('馬番'))}, {int(renka.get('馬番'))}")
    else:
        st.write("出走頭数が少ないため単勝・複勝推奨です。")

with tab3:
    selected_horse = st.selectbox("詳細を見る馬を選択", ranked_df.get("馬名"))
    info = ranked_df.query("馬名 == @selected_horse").squeeze()
    st.subheader(f"{info.get('印')} {int(info.get('馬番'))}番 {info.get('馬名')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("単勝オッズ", f"{info.get('単勝オッズ')}倍")
    c2.metric("🎯 的中スコア", f"{info.get('的中スコア')}点")
    c3.metric("💰 回収スコア", f"{info.get('回収スコア')}pt")
    st.write(f"**騎手 / 斤量：** {info.get('騎手')} / {info.get('斤量')}kg")
    st.write(f"**前走成績：** {int(info.get('前走着順'))}着（上がり3F順位：{int(info.get('前走上がり3F順位'))}位）")

with tab4:
    st.write("リアルタイムのオッズや前走成績を書き換えて再計算できます。")
    edit_cols = ("枠番", "馬番", "馬名", "騎手", "斤量", "単勝オッズ", "前走着順", "前走上がり3F順位", "同コース複勝率", "騎手勝率")
    edited = st.data_editor(curr_df.loc[:, edit_cols], use_container_width=True, hide_index=True)
    if st.button("🔄 AIスコアを再計算する", type="primary"):
        st.session_state[state_key] = edited
        st.rerun()
