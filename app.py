# ===== app.py : 유행 데이터 수집 + 바이럴 동향 예측 =====
# 필요 파일: app.py, train_data.csv (같은 폴더에 둘 것)
# requirements: streamlit, pandas, numpy, scikit-learn, requests, google-api-python-client

import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="유행 동향 예측기", page_icon="🔮", layout="wide")

# =====================================================================
# [예측 파트] 의미 범주 사전
# =====================================================================
FATIGUE  = ['지겹', '질리', '질렸', '물리', '식상', '뇌절', '그만', '노잼', '재미없', '뻔하']
FORCED   = ['억지', '바이럴', '광고', '협찬', '마케팅', '강요', '띄우', '밀어']
REJECT   = ['안사', '안먹', '별로', '맛없', '실망', '싫어', '비싸']
POSITIVE = ['맛있', '존맛', '꿀맛', '먹고싶', '궁금', '사먹', '최고', '대박']
FOODS = ['탕후루', '약과', '마카롱', '뚱카롱', '흑당', '버블티', '요아정', '떡볶이', '티라미수',
         '달고나', '카스테라', '도너츠', '도넛', '붕어빵', '호떡', '와플', '크로플', '마라탕',
         '초콜릿', '두바이', '소금빵', '베이글', '케이크', '쿠키', '아이스크림', '빙수',
         '츄러스', '타르트', '푸딩', '연어', '깍두기', '대창', '버터떡', '두쫀쿠', '밤티',
         '로제', '곱창', '마카', '슈크림', '탕수육', '젤리', '포켓몬', '소떡', '핫도그', '붕어']

CATS = {'f_fatigue': FATIGUE, 'f_forced': FORCED, 'f_reject': REJECT, 'f_positive': POSITIVE}

FEATS = ['rel', 'rel_ma4', 'rel_chg', 'cmt_chg'] + [
    f + s for f in ['r_f_fatigue', 'r_f_forced', 'r_f_reject', 'r_f_positive', 'r_f_alt']
    for s in ['_ma3', '_chg']
]

CAT_LABEL = {'f_forced': '강요감', 'f_alt': '대체재 언급',
             'f_fatigue': '피로감', 'f_reject': '거부', 'f_positive': '긍정'}

STAGE_BASE = {
    '상승기':    {'강요감': 0.46, '대체재 언급': 7.88, '피로감': 0.32, '거부': 1.46, '긍정': 10.46},
    '정점 부근': {'강요감': 1.01, '대체재 언급': 10.76, '피로감': 0.60, '거부': 2.60, '긍정': 11.47},
    '초기 하강': {'강요감': 0.47, '대체재 언급': 9.79, '피로감': 0.41, '거부': 2.41, '긍정': 11.40},
    '후기 소멸': {'강요감': 0.33, '대체재 언급': 8.31, '피로감': 0.35, '거부': 1.54, '긍정': 7.63},
}

PERIOD_OPTIONS = {
    "전체 (예측 권장)": (None, True),
    "최근 2년": (730, True),
    "최근 1년": (365, True),
    "최근 6개월": (182, True),
    "최근 3개월 (12주)": (90, False),
    "최근 8주": (56, False),
    "최근 4주": (28, False),
}


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_data.csv")
    if not os.path.exists(path):
        return None, None
    tr = pd.read_csv(path, encoding="utf-8-sig")
    m = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                   learning_rate=0.05, random_state=42)
    m.fit(tr[FEATS].fillna(0), tr["label"])
    return m, tr


def build_weekly_features(merged, keyword):
    df = merged.copy()
    df = df.dropna(subset=["comment"])
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df["comment"] = df["comment"].astype(str)
    masked = df["comment"].str.replace(keyword, "<PRODUCT>", regex=False)

    df["comment_like_count"] = pd.to_numeric(df.get("comment_like_count", 0), errors="coerce").fillna(0)
    df["view_count"] = pd.to_numeric(df.get("view_count", 0), errors="coerce").fillna(0)
    df["w"] = (1 + np.log1p(df["comment_like_count"])) * (1 + np.log1p(df["view_count"]))

    for col, pats in CATS.items():
        df[col] = masked.apply(lambda t: any(p in t for p in pats))
    others = [f for f in FOODS if f != keyword]
    df["f_alt"] = masked.apply(lambda t: any(o in t for o in others))

    si_src = merged.copy()
    si_src["date"] = pd.to_datetime(si_src["date"], errors="coerce").dt.normalize()
    si = si_src.dropna(subset=["date"]).groupby("date")["search_interest"].first().sort_index()
    si = pd.to_numeric(si, errors="coerce")
    if si.dropna().empty:
        return pd.DataFrame()
    full = pd.date_range(si.index.min(), si.index.max(), freq="D")
    si = si.reindex(full).interpolate().ffill().bfill()

    g = df.groupby("date")
    daily = pd.DataFrame({"n_comments": g.size(), "w_total": g["w"].sum()})
    for c in list(CATS) + ["f_alt"]:
        daily["w_" + c] = df[df[c]].groupby("date")["w"].sum()
    daily = daily.reindex(full, fill_value=0).fillna(0)
    daily
