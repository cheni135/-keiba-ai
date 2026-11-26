import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="自分専用 競馬AI", layout="wide")
st.title("🏇 自分専用 競馬AI予想")

# 開催日リスト（直近の土日を含む前後1週間を生成）
today = datetime.now()
date_options = {}
for i in range(-5, 6):
    d = today + timedelta(days=i)
    if d.weekday() in:
        label = d.strftime("%Y/%m/%d") + ("(土)" if d.weekday() == 5 else "(日)")
        date_options[label] = d.strftime("%Y%m%d")

if not date_options:
    label = today.strftime("%Y/%m/%d")
    date_options[label] = today.strftime("%Y%m%d")

# 1. 開催日の選択
selected_date_label = st.selectbox("開催日を選択", list(date_options.keys()))
kaisai_date = date_options[selected_date_label]

# レース一覧を取得する関数
@st.cache_data(ttl=3600)
def get_race_list(date_str):
    url = f"https://netkeiba.com{date_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    res.encoding = "EUC-JP"
    soup = BeautifulSoup(res.text, "html.parser")
    
    races = {}
    for dl in soup.select(".RaceList_DataList"):
        place_tag = dl.select_one(".RaceList_DataTitle")
        if not place_tag:
            continue
        parts = place_tag.text.strip().split()
        place_name = parts[1] if len(parts) > 1 else place_tag.text.strip()
        races[place_name] = []
        
        for li in dl.select("li.RaceList_DataItem"):
            a_tag = li.select_one("a")
            if not a_tag or "race_id=" not in a_tag.get("href", ""):
                continue
            race_id = re.search(r"race_id=(\d+)", a_tag["href"]).group(1)
            r_num = li.select_one(".Race_Num").text.strip() if li.select_one(".Race_Num") else ""
            r_title = li.select_one(".ItemTitle").text.strip() if li.select_one(".ItemTitle") else ""
            races[place_name].append({"label": f"{r_num} {r_title}", "race_id": race_id})
    return races

# 出馬表を取得する関数
@st.cache_data(ttl=600)
def get_shutuba(race_id):
    url = f"https://netkeiba.com{race_id}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    res.encoding = "EUC-JP"
    soup = BeautifulSoup(res.text, "html.parser")
    
    rows = []
    for tr in soup.select("tr.HorseList"):
        tds = tr.select("td")
        if len(tds) < 11:
            continue
        waku = tds[0].text.strip()
        umaban = tds[1].text.strip()
        horse_name = tds[3].text.strip()
        sex_age = tds[4].text.strip()
        weight = tds[5].text.strip()
        jockey = tds[6].text.strip()
        odds_text = tds[9].text.strip()
        pop_text = tds[10].text.strip()
        
        try:
            odds = float(odds_text)
        except:
            odds = 999.9
        try:
            pop = int(pop_text)
        except:
            pop = 99
            
        rows.append({
            "枠": waku,
            "馬番": int(umaban) if umaban.isdigit() else 0,
            "馬名": horse_name,
            "性齢": sex_age,
            "斤量": weight,
            "騎手": jockey,
            "オッズ": odds,
            "人気": pop
        })
    return pd.DataFrame(rows)

# レース一覧の取得と表示
with st.spinner("レース一覧を取得中..."):
    race_dict = get_race_list(kaisai_date)

if not race_dict:
    st.warning("選択した日付のJRA開催データが見つかりません。別の土日を選んでみてください。")
else:
    col1, col2 = st.columns(2)
    with col1:
        selected_place = st.selectbox("競馬場", list(race_dict.keys()))
    with col2:
        race_items = race_dict[selected_place]
        selected_race_label = st.selectbox("レース", [r["label"] for r in race_items])
        selected_race_id = next(r["race_id"] for r in race_items if r["label"] == selected_race_label)

    # 出馬表の取得
    with st.spinner("出馬表とAIスコアを計算中..."):
        df = get_shutuba(selected_race_id)

    if df.empty:
        st.info("出馬表データがまだ公開されていません（木曜夕方〜金曜に公開されます）。")
    else:
        # 2. スコアモード切替
        mode = st.radio(
            "AIスコア表示モード",
            ["🎯 的中重視モード（3着内確率）", "💰 回収率重視モード（期待値）"],
            horizontal=True
        )

        # 暫定AIスコア計算（次ステップで過去学習モデルに置き換えます）
        df["的中スコア"] = (100 / (df["オッズ"] ** 0.35)).clip(10, 99).round(1)
        df["回収スコア"] = ((df["的中スコア"] * (df["オッズ"] ** 0.25)) * 1.1).clip(10, 150).round(1)

        if "的中" in mode:
            df["表示スコア"] = df["的中スコア"]
        else:
            df["表示スコア"] = df["回収スコア"]

        df = df.sort_values("表示スコア", ascending=False).reset_index(drop=True)
        marks = ["◎", "○", "▲", "△", "☆"]
        df["印"] = [marks[i] if i < len(marks) else "" for i in range(len(df))]

        # 3. タブ切り替え表示
        tab1, tab2, tab3 = st.tabs(["📋 出馬表・AIスコア", "🐎 馬別詳細", "📊 データ分析"])

        with tab1:
            show_df = df[["印", "枠", "馬番", "馬名", "騎手", "オッズ", "人気", "表示スコア"]]
            st.dataframe(show_df, use_container_width=True, hide_index=True)

        with tab2:
            selected_horse = st.selectbox("詳細を見る馬を選択", df["馬名"])
            info = df[df["馬名"] == selected_horse].iloc[0]
            st.subheader(f"{info['印']} {info['馬番']}番 {info['馬名']}")
            st.write(f"**性齢・斤量：** {info['性齢']} / {info['斤量']}kg　｜　**騎手：** {info['騎手']}")
            st.write(f"**単勝オッズ：** {info['オッズ']}倍（{info['人気']}人気）")
            st.metric("AI評価スコア", f"{info['表示スコア']} 点")

        with tab3:
            st.write("枠順別の平均AIスコア")
            waku_df = df.groupby("枠")["表示スコア"].mean().reset_index()
            st.bar_chart(waku_df.set_index("枠"))
