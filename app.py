# ===== app.py : 유행 데이터 수집 + 바이럴 동향 예측 (기간·조회수 필터 추가) =====
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

# 기간 옵션: 표시명 -> (일수, 예측 신뢰 가능 여부)
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
    daily["search"] = si

    wk = daily.resample("W-MON").sum()
    wk["search"] = daily["search"].resample("W-MON").mean()
    for c in list(CATS) + ["f_alt"]:
        wk["r_" + c] = np.where(wk["w_total"] > 0, wk["w_" + c] / wk["w_total"] * 100, np.nan)

    wk = wk[wk["n_comments"] >= 5].copy()
    if len(wk) < 3:
        return pd.DataFrame()

    peak_val = wk["search"].max()
    wk["rel"] = wk["search"] / peak_val * 100 if peak_val > 0 else 0
    wk["rel_ma4"] = wk["rel"].rolling(4, min_periods=1).mean()
    wk["rel_chg"] = wk["rel"].pct_change(4).replace([np.inf, -np.inf], np.nan)
    wk["cmt_chg"] = wk["n_comments"].pct_change(4).replace([np.inf, -np.inf], np.nan)
    for c in list(CATS) + ["f_alt"]:
        wk["r_" + c + "_ma3"] = wk["r_" + c].rolling(3, min_periods=1).mean()
        wk["r_" + c + "_chg"] = wk["r_" + c + "_ma3"].diff(4)

    wk["weeks_from_peak"] = ((wk.index - wk["search"].idxmax()).days / 7).astype(int)
    return wk


def judge_stage(row, recent_trend=0.0):
    w, r = row["weeks_from_peak"], row["rel"]
    if recent_trend > 8 and r >= 15:
        return "상승기"
    if w < 0:
        return "상승기"
    if w <= 4:
        return "정점 부근"
    if r >= 30:
        return "초기 하강"
    return "후기 소멸"


def recent_trend_of(wk, weeks=8):
    s = wk["rel"].dropna()
    if len(s) < 3:
        return 0.0
    tail = s.tail(weeks)
    return float(tail.iloc[-1] - tail.iloc[0])


STAGE_MSG = {
    "상승기": "아직 정점 전입니다. 유행이 커지는 중이라 재료 확보 여력이 있지만, "
              "**강요감과 대체재 언급이 함께 뛰기 시작하면** 고비가 가깝다는 신호입니다.",
    "정점 부근": "**가장 주의해야 할 구간**입니다. 학습한 14개 품목 평균으로 이 시점에서 "
                 "강요감이 상승기의 약 2.2배로 올랐습니다. 대량 발주는 이 시점부터 위험합니다.",
    "초기 하강": "정점을 지나 내려오는 중입니다. 학습한 품목들은 이 구간에서 "
                 "**평균 4~8주에 걸쳐** 정점 대비 30% 아래로 떨어졌습니다. 재고 소진 계획이 필요합니다.",
    "후기 소멸": "이미 많이 내려온 상태입니다. 남은 하락 폭이 작아 급락 위험은 줄었지만, "
                 "**이 수준에서 정착할지 완전히 사라질지**는 품목마다 갈렸습니다.",
}


# =====================================================================
# [수집 파트]
# =====================================================================
def crawl_comments(keyword, api_key, max_videos=100, max_comments=100,
                   period_days=None, sort_by_views=False,
                   top_by_views=None, min_views=0):
    """period_days: 기간 필터 / sort_by_views: 조회수순 검색
       top_by_views: 조회수 상위 N개 영상만 사용 / min_views: 최소 조회수"""
    yt = build("youtube", "v3", developerKey=api_key)
    published_after = None
    if period_days:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=period_days)
        published_after = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

    ids, seen, page = [], set(), None
    while len(ids) < max_videos:
        try:
            params = dict(q=keyword, part="id", type="video",
                          maxResults=min(50, max_videos - len(ids)), pageToken=page,
                          regionCode="KR", relevanceLanguage="ko",
                          order="viewCount" if sort_by_views else "relevance")
            if published_after:
                params["publishedAfter"] = published_after
            res = yt.search().list(**params).execute()
        except HttpError as e:
            st.error(f"유튜브 검색 실패: {e}")
            st.info("대부분 '오늘 할당량 초과'입니다. 한국시간 오후 4~5시경 리셋되니 그 후에 다시, 또는 영상 수를 줄여보세요.")
            return pd.DataFrame()
        for it in res.get("items", []):
            v = it["id"]["videoId"]
            if v not in seen:
                seen.add(v); ids.append(v)
        page = res.get("nextPageToken")
        if not page:
            break

    # 영상 메타데이터
    details = {}
    for i in range(0, len(ids), 50):
        try:
            res = yt.videos().list(part="snippet,statistics",
                                   id=",".join(ids[i:i+50])).execute()
        except HttpError:
            continue
        for it in res.get("items", []):
            s, stt = it["snippet"], it.get("statistics", {})
            details[it["id"]] = {
                "video_title": s["title"], "channel_title": s["channelTitle"],
                "view_count": int(stt["viewCount"]) if "viewCount" in stt else None,
                "video_published_at": s["publishedAt"]}

    # --- 조회수 필터: 최소 조회수 미만 제외 + 상위 N개만 ---
    if details:
        meta_df = pd.DataFrame.from_dict(details, orient="index")
        meta_df.index.name = "video_id"; meta_df = meta_df.reset_index()
        meta_df["view_count"] = pd.to_numeric(meta_df["view_count"], errors="coerce").fillna(0)
        before = len(meta_df)
        if min_views > 0:
            meta_df = meta_df[meta_df["view_count"] >= min_views]
        meta_df = meta_df.sort_values("view_count", ascending=False)
        if top_by_views:
            meta_df = meta_df.head(top_by_views)
        kept = set(meta_df["video_id"])
        ids = [v for v in ids if v in kept]
        details = {k: v for k, v in details.items() if k in kept}
        if before != len(ids):
            st.caption(f"영상 {before}개 중 조회수 기준으로 {len(ids)}개 선별")

    if not ids:
        st.warning("조회수 조건을 만족하는 영상이 없습니다. 최소 조회수를 낮춰보세요.")
        return pd.DataFrame()

    records = []
    prog = st.progress(0.0, text="유튜브 댓글 수집 중...")
    for i, v in enumerate(ids, 1):
        page = None; got = 0
        while got < max_comments:
            try:
                res = yt.commentThreads().list(part="snippet", videoId=v,
                    maxResults=100, pageToken=page, textFormat="plainText",
                    order="relevance").execute()
            except HttpError:
                break
            for it in res.get("items", []):
                c = it["snippet"]["topLevelComment"]["snippet"]
                records.append({"video_id": v, "comment": c["textOriginal"],
                    "comment_published_at": c["publishedAt"],
                    "comment_like_count": c.get("likeCount", 0)})
                got += 1
            page = res.get("nextPageToken")
            if not page:
                break
            time.sleep(0.1)
        prog.progress(i / max(1, len(ids)), text=f"유튜브 댓글 수집 중... ({i}/{len(ids)})")
    prog.empty()

    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    meta = pd.DataFrame.from_dict(details, orient="index")
    meta.index.name = "video_id"; meta = meta.reset_index()
    df = df.merge(meta, on="video_id", how="left")
    df.drop_duplicates(subset=["video_id", "comment"], inplace=True)
    df["comment_published_at"] = pd.to_datetime(df["comment_published_at"], utc=True, errors="coerce")

    if period_days:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=period_days)
        df = df[df["comment_published_at"] >= cutoff]
        if df.empty:
            return pd.DataFrame()

    df["comment_year_month"] = df["comment_published_at"].dt.tz_convert(None).dt.to_period("M").astype(str)
    df.sort_values("comment_published_at", inplace=True)
    return df.reset_index(drop=True)


def get_search_trend(keyword, start, end, serpapi_key):
    if not serpapi_key:
        st.info("검색량 추이는 SerpApi 키가 있어야 나옵니다. (사이드바에 입력)")
        return pd.DataFrame()
    try:
        params = {
            "engine": "google_trends", "q": keyword, "data_type": "TIMESERIES",
            "date": f"{start} {end}", "geo": "KR", "hl": "ko", "api_key": serpapi_key,
        }
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
        data = r.json()
        if "error" in data:
            st.warning(f"검색량 수집 오류: {data['error']}")
            return pd.DataFrame()
        timeline = data.get("interest_over_time", {}).get("timeline_data", [])
        rows = []
        for pt in timeline:
            ts = pt.get("timestamp")
            vals = pt.get("values", [])
            val = vals[0].get("extracted_value", 0) if vals else 0
            d = pd.to_datetime(int(ts), unit="s").date() if ts else pt.get("date")
            rows.append({"date": d, "search_interest": val})
        return pd.DataFrame(rows)
    except Exception as e:
        st.warning(f"검색량 추이 수집 실패: {e}")
        return pd.DataFrame()


def merge_files(df_comments, df_trend):
    c = df_comments.copy()
    c["date"] = pd.to_datetime(c["comment_published_at"], utc=True, errors="coerce") \
                  .dt.tz_localize(None).dt.normalize()
    if df_trend.empty:
        c["search_interest"] = pd.NA
        return c.sort_values("date").reset_index(drop=True)
    t = df_trend.copy()
    t["date"] = pd.to_datetime(t["date"], errors="coerce").dt.normalize()
    t = t.dropna(subset=["date"])
    c2 = c.dropna(subset=["date"]).copy()
    c2["_key"] = c2["date"].map(lambda x: x.toordinal())
    t2 = t.copy()
    t2["_key"] = t2["date"].map(lambda x: x.toordinal())
    left = c2.sort_values("_key")
    right = t2[["_key", "search_interest"]].sort_values("_key")
    m1 = pd.merge_asof(left, right, on="_key", direction="nearest").drop(columns="_key")
    comment_keys = set(c2["_key"])
    extra = t2[~t2["_key"].isin(comment_keys)][["date", "search_interest"]].copy()
    combined = pd.concat([m1, extra], ignore_index=True)
    return combined.sort_values("date").reset_index(drop=True)


# =====================================================================
# 화면
# =====================================================================
st.title("🔮 유행 동향 예측기")
st.caption("품목을 넣으면 유튜브 댓글과 검색량을 모아, 지금 유행이 어느 국면에 있는지 읽어냅니다.")

model, train_ref = load_model()

try:
    yt_key = st.secrets["YOUTUBE_API_KEY"]
except Exception:
    yt_key = ""
try:
    serp_key = st.secrets["SERPAPI_KEY"]
except Exception:
    serp_key = ""

with st.sidebar:
    st.header("설정")
    if not yt_key:
        yt_key = st.text_input("YouTube API 키", type="password")
    if not serp_key:
        serp_key = st.text_input("SerpApi 키", type="password")

    st.subheader("① 기간")
    period_label = st.selectbox("수집 기간", list(PERIOD_OPTIONS.keys()), index=0)
    period_days, pred_ok = PERIOD_OPTIONS[period_label]
    if not pred_ok:
        st.warning("이 기간은 짧아 **예측이 부정확하거나 불가**합니다. "
                   "예측이 목적이면 '전체' 또는 '최근 1년' 이상을 쓰세요.")

    st.subheader("② 영상 선별")
    max_videos = st.slider("검색할 영상 수", 10, 500, 300, step=10,
                           help="넉넉히 모은 뒤 아래 조건으로 걸러냅니다.")
    use_top = st.checkbox("조회수 상위 N개만 분석", value=True)
    top_by_views = st.slider("상위 N개", 10, 300, 100, step=10) if use_top else None
    min_views = st.number_input("최소 조회수 (미만 제외)", min_value=0, value=1000, step=500)
    sort_by_views = st.checkbox("조회수순으로 검색", value=False,
                                help="켜면 조회수 높은 영상 위주로 찾지만, 오래된 영상에 치우칠 수 있습니다.")

    st.subheader("③ 댓글")
    max_comments = st.slider("영상당 댓글 수", 20, 10000, 100, step=10)

    st.divider()
    if model is None:
        st.error("train_data.csv 가 없어 예측을 쓸 수 없습니다. app.py와 같은 폴더에 두세요.")
    else:
        st.success(f"예측 모델 준비됨 (14개 품목 {len(train_ref):,}주 학습)")
        st.caption("품목 교차검증 AUC 0.661 — 프로토타입 단계")

col1, col2 = st.columns([4, 1])
keyword = col1.text_input("품목", placeholder="예: 탕후루", label_visibility="collapsed")
run = col2.button("분석", type="primary", use_container_width=True)

if run:
    if not yt_key:
        st.error("YouTube API 키를 입력하세요. (사이드바)")
    elif not keyword.strip():
        st.warning("품목을 입력하세요.")
    else:
        kw = keyword.strip()
        with st.spinner(f"'{kw}' 유튜브 댓글 수집 중... ({period_label}, 몇 분 걸릴 수 있습니다)"):
            df = crawl_comments(kw, yt_key, max_videos, max_comments,
                                period_days, sort_by_views, top_by_views, int(min_views))
        if df.empty:
            st.error("수집된 댓글이 없습니다. 품목명·API 키·기간·조회수 조건을 확인해 주세요.")
        else:
            # 댓글 기간과 무관하게 검색량은 최근 1년치를 수집 (추세 비교 기준 확보)
            trend_end = pd.Timestamp.now(tz="UTC")
            trend_start = trend_end - pd.Timedelta(days=365)
            start = trend_start.strftime("%Y-%m-%d")
            end   = trend_end.strftime("%Y-%m-%d")
            with st.spinner("구글 검색량 추이 수집 중... (최근 1년)"):
                trend = get_search_trend(kw, start, end, serp_key)

            c1, c2, c3 = st.columns(3)
            c1.metric("수집 댓글", f"{len(df):,}")
            c2.metric("분석 영상 수", f"{df['video_id'].nunique():,}")
            c3.metric("댓글 기간", f"{df['comment_published_at'].min().date()} ~ {df['comment_published_at'].max().date()}")

            merged = merge_files(df, trend)
            # ============ 예측 ============
            st.subheader("🔮 바이럴 동향 예측")
            if not pred_ok:
                st.warning(f"'{period_label}'은 기간이 짧아 예측을 건너뜁니다. "
                           "아래 수집 데이터는 정상적으로 사용할 수 있습니다.")
            elif trend.empty:
                st.warning("검색량 데이터가 없어 예측을 할 수 없습니다. SerpApi 키를 확인해 주세요.")
            elif model is None:
                st.warning("train_data.csv 가 없어 예측을 할 수 없습니다.")
            else:
                wk = build_weekly_features(merged, kw)
                if wk.empty:
                    st.warning("주간 데이터가 부족해 예측할 수 없습니다. 기간이나 영상 수를 늘려 보세요.")
                else:
                    last = wk.iloc[-1]
                    stage = judge_stage(last, recent_trend_of(wk))
                    X = wk[FEATS].fillna(0).iloc[[-1]]
                    prob = float(model.predict_proba(X)[0, 1])

                    a, b = st.columns([1, 2])
                    a.metric("현재 국면", stage)
                    b.metric("4주 내 15% 이상 하락 가능성", f"{prob*100:.0f}%")
                    st.progress(min(max(prob, 0.0), 1.0))
                    st.info(STAGE_MSG[stage])

                    st.markdown("**댓글 신호 — 14개 품목 같은 국면 평균과 비교**")
                    base = STAGE_BASE[stage]
                    cols = st.columns(5)
                    for i, (col_key, label) in enumerate(CAT_LABEL.items()):
                        val = float(wk["r_" + col_key + "_ma3"].iloc[-1])
                        ref = base[label]
                        diff = val - ref
                        cols[i].metric(label, f"{val:.2f}%", f"{diff:+.2f}%p")
                    st.caption("위 화살표는 같은 국면에 있던 14개 품목 평균과의 차이입니다. "
                               "강요감·대체재 언급이 평균보다 크게 높으면 고비가 빨리 올 수 있습니다.")

                    with st.expander("주간 분석 데이터 보기"):
                        show = wk[["search", "rel", "n_comments", "r_f_forced_ma3",
                                   "r_f_alt_ma3", "r_f_fatigue_ma3", "weeks_from_peak"]].copy()
                        show.columns = ["검색량", "정점대비(%)", "댓글수", "강요감(%)",
                                        "대체재(%)", "피로감(%)", "정점기준주차"]
                        st.dataframe(show.round(2), use_container_width=True)

                    st.warning("이 예측은 14개 품목으로 학습한 **프로토타입**입니다. "
                               "품목 교차검증 AUC 0.661로 무작위보다는 낫지만 실무 배포 기준에는 미치지 못하며, "
                               "품목에 따라 편차가 큽니다. 발주 판단의 유일한 근거로 삼지 마세요.")

            # ============ 수집 데이터 ============
            st.subheader("① 유튜브 댓글")
            st.dataframe(df, use_container_width=True, height=240)
            st.download_button("댓글 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{kw}_comments.csv", mime="text/csv")

            st.subheader("② 구글 검색량 추이 (0~100)")
            if trend.empty:
                st.info("검색량 추이를 가져오지 못했습니다. SerpApi 키와 사용량을 확인해 주세요.")
            else:
                st.line_chart(trend.set_index("date")["search_interest"])
                st.download_button("검색량 CSV", trend.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"{kw}_search_trend.csv", mime="text/csv")

            st.subheader("③ 합친 파일 (댓글 + 그 시점 검색량)")
            st.dataframe(merged, use_container_width=True, height=240)
            st.download_button("⭐ 합친 CSV 다운로드",
                               merged.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{kw}_merged.csv", mime="text/csv")
else:
    st.info("품목을 넣고 '분석'을 누르세요.")
