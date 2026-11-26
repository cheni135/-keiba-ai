import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta

st.set_page_config(page_title="自分専用 競馬AI", layout="wide")
st.title("🏇 自分専用 競馬AI予想")

# 安全にWebデータを取得する関数（ブロック時はエラーにせず空データを返す）
@st.cache_data(ttl=3600)
def safe_get_html(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
    }
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.encoding = "EUC-JP"
        return res.text
    except Exception:
        return ""

# 直近の土日を生成
today = datetime.now()
date_options = {}
for i in range(-5, 6):
    d = today + timedelta(days=i)
    if d.weekday() == 5 or d.weekday() == 6:
        label = d.strftime("%Y/%m/%d") + ("(土)" if d.weekday() == 5 else "(日)")
        date_options[label] = d.strftime("%Y%m%d")

selected_date_label = st.selectbox("開催日を選択", list(date_options.keys()))
kaisai_date = date_options.get(selected_date_label)

html = safe_get_html(f"https://netkeiba.com{kaisai_date}")

if not html:
    st.warning("⚠️ サーバーからの直接取得が制限されています。Google Colabで作成したデータを読み込む方式に切り替えます。")
    
    # サンプル表示（Colab連携用テスト）
    sample_data = pd.DataFrame({
        "印": ("◎", "○", "▲", "△"),
        "枠": ("1", "2", "3", "4"),
        "馬番": (1, 2, 3, 4),
        "馬名": ("サンプルホースA", "サンプルホースB", "サンプルホースC", "サンプルホースD"),
        "騎手": ("ルメール", "川田将雅", "武豊", "戸崎圭太"),
        "オッズ": (2.8, 5.4, 14.2, 8.9),
        "的中スコア": (88.5, 81.2, 64.0, 72.5),
        "回収スコア": (95.0, 108.4, 132.0, 112.1)
    })
    
    mode = st.radio("AIスコア表示モード", ("🎯 的中重視モード", "💰 回収率重視モード"), horizontal=True)
    score_col = "的中スコア" if "的中" in mode else "回収スコア"
    sample_data = sample_data.sort_values(score_col, ascending=False)
    st.dataframe(sample_data, use_container_width=True, hide_index=True)
else:
    st.success("✅ データ取得に成功しました！")
