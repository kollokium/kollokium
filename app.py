# ===== app.py : 유행 데이터 수집 + 바이럴 동향 예측 =====
# 필요 파일: app.py, train_data.csv (같은 폴더에 둘 것)
# requirements: streamlit, pandas, numpy, scipy, scikit-learn, requests, google-api-python-client

import os
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="유행 동향 예측기", page_icon="🔮", layout="wide")

# =====================================================================
# [예측 파트] 확산 동역학 지표
# =====================================================================
FEATS = ['rel', 'rel_ma4', 'rel_chg4', 'eff_per_video_s', 'log_lag', 'gini_s',
         'n_channels_s', 'n_new_videos_s', 'growth_s', 'accel_s',
         'eff_per_video_s_d4', 'log_lag_d4', 'n_channels_s_d4',
         'n_new_videos_s_d4', 'gini_s_d4', 'bass_r2', 'log_qp']

STAGE_REF = {
    '상승기':    {'반응효율': 25.44, '반응지연': 3194.5, '집중도': 0.48, '참여채널': 7.6, '신규영상': 1.0},
    '정점 부근': {'반응효율': 28.70, '반응지연': 428.0, '집중도': 0.65, '참여채널': 36.1, '신규영상': 11.6},
    '초기 하강': {'반응효율': 8.46, '반응지연': 3015.3, '집중도': 0.57, '참여채널': 19.7, '신규영상': 1.6},
    '후기 소멸': {'반응효율': 5.26, '반응지연': 6232.6, '집중도': 0.54, '참여채널': 25.3, '신규영상': 1.1},
}

STAGE_MSG = {
    "상승기": "아직 정점 전입니다. 확산이 커지는 중이라 재료 확보 여력이 있습니다. "
              "다만 **참여 채널 수가 급증하고 반응 지연이 짧아지면** 정점이 임박했다는 신호이니 "
              "그 시점부터 발주량을 조절하세요.",
    "정점 부근": "**가장 주의해야 할 구간**입니다. 학습한 14개 품목 평균으로 이 시점의 참여 채널 수는 "
                 "상승기의 약 4.7배, 반응 지연은 약 7분의 1로 짧아집니다. 열기가 최고조라는 뜻이지만 "
                 "동시에 하락이 시작되는 지점입니다. 대량 발주는 이 시점부터 위험합니다.",
    "초기 하강": "정점을 지나 내려오는 중입니다. 이 구간에서 영상 1편당 댓글 수가 정점 대비 약 3분의 1로 "
                 "떨어집니다. **영상은 계속 올라오는데 반응이 안 붙는 상태**이므로 재고 소진 계획이 필요합니다.",
    "후기 소멸": "이미 많이 내려온 상태입니다. 남은 하락 폭이 작아 급락 위험은 줄었지만 "
                 "**이 수준에서 정착할지 완전히 사라질지**는 품목마다 갈렸습니다. "
                 "명절·계절 요인이 있는 품목은 재상승할 수 있습니다.",
}

PERIOD_OPTIONS = {
    "전체 (예측 권장)": (None, True, 1825),
    "최근 2년": (730, True, 1095),
    "최근 1년": (365, True, 730),
    "최근 6개월": (182, True, 365),
    "최근 3개월 (12주)": (90, False, 365),
    "최근 8주": (56, False, 365),
    "최근 4주": (28, False, 365),
}

QUERY_SUFFIXES = ["", " 먹방", " 리뷰", " 후기", " 맛집", " 유행", " 만들기",
                  " 브이로그", " asmr", " 추천", " 존맛", " 내돈내산"]


@st.cache_resource
def load_model():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_data.csv")
    if not os.path.exists(path):
        return None, None
    tr = pd.read_csv(path, encoding="utf-8-sig")
    m = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                   learning_rate=0.05, random_state=42)
    m.fit(tr[FEATS].fillna(0), tr["label"])
    return m, tr


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return (2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())


def bass_curve(t, p, q, M):
    e = np.exp(-(p + q) * t)
    return M * ((p + q) ** 2 / p) * e / (1 + (q / p) * e) ** 2


def fit_bass(weekly_counts):
    y = np.asarray(weekly_counts, dtype=float)
    if len(y) < 15 or y.sum() == 0:
        return 0.0, 0.0
    t = np.arange(len(y), dtype=float)
    try:
        popt, _ = curve_fit(bass_curve, t, y, p0=[0.01, 0.3, y.sum()], maxfev=20000,
                            bounds=([1e-6, 1e-4, y.sum() * 0.5],
                                    [0.5, 2.0, y.sum() * 50]))
        p_, q_, _ = popt
        pred = bass_curve(t, *popt)
        denom = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ((y - pred) ** 2).sum() / denom if denom > 0 else 0.0
        return max(float(r2), 0.0), float(q_ / p_)
    except Exception:
        return 0.0, 0.0


def build_weekly_features(merged):
    df = merged.copy()
    df["ct"] = pd.to_datetime(df.get("comment_published_at"), utc=True,
                              errors="coerce").dt.tz_localize(None)
    df["vt"] = pd.to_datetime(df.get("video_published_at"), utc=True,
                              errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["ct"])
    if df.empty:
        return pd.DataFrame(), (0.0, 0.0)
    df["date"] = df["ct"].dt.normalize()

    df["lag_h"] = (df["ct"] - df["vt"]).dt.total_seconds() / 3600
    df.loc[(df["lag_h"] < 0) | (df["lag_h"] > 24 * 365 * 3), "lag_h"] = np.nan

    si = pd.to_numeric(df.groupby("date")["search_interest"].first().sort_index(),
                       errors="coerce")
    if si.dropna().empty:
        return pd.DataFrame(), (0.0, 0.0)
    full = pd.date_range(si.index.min(), si.index.max(), freq="D")
    si = si.reindex(full).interpolate().ffill().bfill()

    W = "W-MON"
    g = df.set_index("ct")
    wk = pd.DataFrame({"n_comments": g.resample(W).size()})
    wk["n_videos_active"] = g.resample(W)["video_id"].nunique()
    wk["n_channels"] = g.resample(W)["channel_title"].nunique()
    wk["lag_median"] = g.resample(W)["lag_h"].median()

    vids = df.dropna(subset=["vt"]).drop_duplicates("video_id")[["video_id", "vt"]]
    wk["n_new_videos"] = vids.set_index("vt").resample(W).size().reindex(wk.index, fill_value=0)
    wk["gini"] = g.resample(W)["video_id"].apply(
        lambda s: gini(s.value_counts().values) if len(s) > 1 else np.nan)
    wk["search"] = si.resample(W).mean().reindex(wk.index)

    wk = wk[wk["n_comments"] >= 20].copy()
    if len(wk) < 10:
        return pd.DataFrame(), (0.0, 0.0)

    wk["eff_per_video"] = wk["n_comments"] / wk["n_videos_active"].replace(0, np.nan)
    peak_val = wk["search"].max()
    wk["rel"] = wk["search"] / peak_val * 100 if peak_val > 0 else 0

    for c in ["eff_per_video", "lag_median", "gini", "n_channels", "n_new_videos", "n_comments"]:
        wk[c + "_s"] = wk[c].rolling(3, min_periods=1, center=True).mean()
    wk["log_lag"] = np.log1p(wk["lag_median_s"])
    wk["growth_s"] = wk["n_comments_s"].pct_change(2).replace([np.inf, -np.inf], np.nan)
    wk["accel_s"] = wk["growth_s"].diff(2)
    for c in ["eff_per_video_s", "log_lag", "n_channels_s", "n_new_videos_s", "gini_s"]:
        wk[c + "_d4"] = wk[c].pct_change(4).replace([np.inf, -np.inf], np.nan)
    wk["rel_ma4"] = wk["rel"].rolling(4, min_periods=1).mean()
    wk["rel_chg4"] = wk["rel"].pct_change(4).replace([np.inf, -np.inf], np.nan)

    bass_r2, qp = fit_bass(wk["n_comments"].values)
    wk["bass_r2"] = bass_r2
    wk["log_qp"] = np.log1p(qp)

    wk["weeks_from_peak"] = ((wk.index - wk["search"].idxmax()).days / 7).astype(int)
    return wk, (bass_r2, qp)


def recent_trend_of(wk, weeks=8):
    s = wk["rel"].dropna()
    if len(s) < 3:
        return 0.0
    tail = s.tail(weeks)
    return float(tail.iloc[-1] - tail.iloc[0])


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


def interpret_bass(r2, qp):
    if r2 < 0.2:
        return ("반복 유행형", "단일 확산 곡선으로 설명되지 않습니다. 명절·계절처럼 "
                            "주기적으로 다시 뜨는 품목일 가능성이 있습니다.")
    if qp > 50:
        return ("입소문 주도형", "외부 노출보다 사람 간 전파가 확산을 이끌었습니다. "
                              "빠르게 퍼지지만 그만큼 빨리 식는 경향이 있습니다.")
    if qp > 10:
        return ("균형형", "초기 노출과 입소문이 함께 확산을 이끌었습니다.")
    return ("초기 노출 주도형", "미디어·광고 등 초기 노출이 확산을 이끌었습니다. "
                            "노출이 끊기면 빠르게 사그라들 수 있습니다.")


# =====================================================================
# [수집 파트]
# =====================================================================
def search_video_ids(yt, keyword, target, published_after, sort_by_views):
    """변형 검색어를 순회하며 후보 영상 ID를 target개까지 모은다."""
    ids, seen = [], set()
    status = st.empty()
    for suf in QUERY_SUFFIXES:
        if len(ids) >= target:
            break
        q = keyword + suf
        page = None
        while len(ids) < target:
            try:
                params = dict(q=q, part="id", type="video", maxResults=50,
                              pageToken=page, regionCode="KR", relevanceLanguage="ko",
                              order="viewCount" if sort_by_views else "relevance")
                if published_after:
                    params["publishedAfter"] = published_after
                res = yt.search().list(**params).execute()
            except HttpError as e:
                st.error(f"유튜브 검색 실패: {e}")
                st.info("대부분 '오늘 할당량 초과'입니다. 한국시간 오후 4~5시경 리셋됩니다.")
                return ids
            for it in res.get("items", []):
                v = it.get("id", {}).get("videoId")
                if v and v not in seen:
                    seen.add(v); ids.append(v)
            page = res.get("nextPageToken")
            if not page:
                break
        status.caption(f"검색 중… '{q}' 까지 후보 {len(ids)}개")
    status.empty()
    return ids[:target]


def crawl_comments(keyword, api_key, target_videos=100, max_comments=100,
                   period_days=None, sort_by_views=False, min_views=0):
    """댓글이 실제로 수집된 영상이 target_videos개가 될 때까지 진행한다.
    댓글이 꺼졌거나 0개인 영상은 세지 않고 건너뛴다."""
    yt = build("youtube", "v3", developerKey=api_key)
    published_after = None
    if period_days:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=period_days)
        published_after = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 댓글 없는 영상까지 감안해 후보를 넉넉히(4배) 확보
    search_target = min(target_videos * 4, 500)
    ids = search_video_ids(yt, keyword, search_target, published_after, sort_by_views)
    if not ids:
        return pd.DataFrame()

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

    # 최소 조회수 필터 (목표를 못 채울 정도면 완화) + 조회수 높은 순으로 시도
    if details:
        meta_df = pd.DataFrame.from_dict(details, orient="index")
        meta_df.index.name = "video_id"; meta_df = meta_df.reset_index()
        meta_df["view_count"] = pd.to_numeric(meta_df["view_count"], errors="coerce").fillna(0)
        found = len(meta_df)
        filtered = meta_df[meta_df["view_count"] >= min_views] if min_views > 0 else meta_df
        if len(filtered) < target_videos:
            filtered = meta_df
        order = filtered.sort_values("view_count", ascending=False)["video_id"].tolist()
        ids = order
        st.caption(f"후보 {found}개 확보 → 댓글 있는 영상 {target_videos}개를 채울 때까지 수집합니다.")

    if not ids:
        return pd.DataFrame()

    records = []
    used_videos, skipped = 0, 0
    prog = st.progress(0.0, text="유튜브 댓글 수집 중...")
    for v in ids:
        if used_videos >= target_videos:
            break
        page = None; got = 0
        vid_records = []
        while got < max_comments:
            try:
                res = yt.commentThreads().list(part="snippet", videoId=v,
                    maxResults=100, pageToken=page, textFormat="plainText",
                    order="relevance").execute()
            except HttpError:
                break
            for it in res.get("items", []):
                c = it["snippet"]["topLevelComment"]["snippet"]
                vid_records.append({"video_id": v, "comment": c["textOriginal"],
                    "comment_published_at": c["publishedAt"],
                    "comment_like_count": c.get("likeCount", 0)})
                got += 1
            page = res.get("nextPageToken")
            if not page:
                break
            time.sleep(0.1)
        if not vid_records:          # 댓글 없음 → 목표에 포함하지 않고 건너뜀
            skipped += 1
            continue
        records.extend(vid_records)
        used_videos += 1
        prog.progress(min(used_videos / target_videos, 1.0),
                      text=f"댓글 수집 중... ({used_videos}/{target_videos}개 영상, 건너뜀 {skipped})")
    prog.empty()
    if skipped:
        st.caption(f"댓글이 없거나 사용중지된 영상 {skipped}개는 건너뛰었습니다.")

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
st.caption("품목을 넣으면 유튜브 댓글과 검색량을 모아, 확산의 시간 구조로 지금 유행이 어느 국면에 있는지 읽어냅니다.")

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
    period_label = st.selectbox("영상·댓글 수집 기간", list(PERIOD_OPTIONS.keys()), index=0)
    period_days, pred_ok, trend_days = PERIOD_OPTIONS[period_label]
    st.caption(f"검색량 추이는 확산 국면 판정을 위해 최근 {trend_days}일까지 수집합니다.")
    if not pred_ok:
        st.warning("이 기간은 짧아 **예측이 부정확하거나 불가**할 수 있습니다. "
                   "확산 지표는 최소 10주 이상의 데이터가 필요합니다.")

    st.subheader("② 영상")
    target_videos = st.slider("분석할 영상 수 (댓글 있는 영상 기준)", 10, 300, 100, step=10,
                              help="댓글이 없는 영상은 세지 않고 건너뜁니다.")
    min_views = st.number_input("최소 조회수 (미만 제외)", min_value=0, value=0, step=500,
                                help="이 조건으로 목표를 못 채우면 개수 확보를 우선합니다.")
    sort_by_views = st.checkbox("조회수순으로 검색", value=False)
    if min_views > 0:
        st.caption("⚠️ 조회수 필터를 강하게 걸면 참여 채널 수·신규 영상 수가 실제보다 적게 잡혀 "
                   "예측이 보수적으로 나올 수 있습니다. (학습 데이터는 필터 없이 만들어졌습니다)")

    st.subheader("③ 댓글")
    max_comments = st.slider("영상당 댓글 수", 20, 10000, 100, step=10)

    st.divider()
    if model is None:
        st.error("train_data.csv 가 없어 예측을 쓸 수 없습니다. app.py와 같은 폴더에 두세요.")
    else:
        st.success(f"예측 모델 준비됨 (14개 품목 {len(train_ref):,}주 학습)")
        st.caption("품목 교차검증 AUC 0.716 (검색량만 쓸 때 0.640)")

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
        with st.spinner(f"'{kw}' 영상·댓글 수집 중... ({period_label}, 몇 분 걸릴 수 있습니다)"):
            df = crawl_comments(kw, yt_key, target_videos, max_comments,
                                period_days, sort_by_views, int(min_views))
        if df.empty:
            st.error("수집된 댓글이 없습니다. 품목명·API 키·기간을 확인해 주세요.")
        else:
            trend_end = pd.Timestamp.now(tz="UTC")
            trend_start = trend_end - pd.Timedelta(days=trend_days)
            cmt_start = df["comment_published_at"].min()
            if pd.notna(cmt_start) and cmt_start < trend_start:
                trend_start = cmt_start
            start = trend_start.strftime("%Y-%m-%d")
            end   = trend_end.strftime("%Y-%m-%d")
            with st.spinner(f"구글 검색량 추이 수집 중... ({start} ~ {end})"):
                trend = get_search_trend(kw, start, end, serp_key)
            merged = merge_files(df, trend)

            c1, c2, c3 = st.columns(3)
            c1.metric("수집 댓글", f"{len(df):,}")
            c2.metric("분석 영상 수", f"{df['video_id'].nunique():,}")
            c3.metric("댓글 기간", f"{df['comment_published_at'].min().date()} ~ {df['comment_published_at'].max().date()}")

            st.subheader("🔮 바이럴 동향 예측")
            if not pred_ok:
                st.warning(f"'{period_label}'은 기간이 짧아 예측을 건너뜁니다. "
                           "아래 수집 데이터는 정상적으로 사용할 수 있습니다.")
            elif trend.empty:
                st.warning("검색량 데이터가 없어 예측을 할 수 없습니다. SerpApi 키를 확인해 주세요.")
            elif model is None:
                st.warning("train_data.csv 가 없어 예측을 할 수 없습니다.")
            else:
                wk, (bass_r2, qp) = build_weekly_features(merged)
                if wk.empty:
                    st.warning("주간 데이터가 부족해 예측할 수 없습니다. "
                               "(댓글 20개 이상인 주가 10주 이상 필요) "
                               "기간을 '전체'로 하거나 영상 수를 늘려 보세요.")
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

                    st.markdown("**확산 지표 — 14개 품목의 같은 국면 평균과 비교**")
                    ref = STAGE_REF[stage]
                    cur = {
                        '반응효율': float(last['eff_per_video_s']),
                        '반응지연': float(last['lag_median_s']),
                        '집중도': float(last['gini_s']),
                        '참여채널': float(last['n_channels_s']),
                        '신규영상': float(last['n_new_videos_s']),
                    }
                    unit = {'반응효율': '개/영상', '반응지연': '시간', '집중도': '',
                            '참여채널': '개', '신규영상': '개'}
                    cols = st.columns(5)
                    for i, k in enumerate(['참여채널', '반응지연', '반응효율', '신규영상', '집중도']):
                        v, r = cur[k], ref[k]
                        if np.isnan(v):
                            cols[i].metric(k, "-")
                            continue
                        fmt = f"{v:.2f}" if k == '집중도' else f"{v:,.0f}"
                        cols[i].metric(k + (f" ({unit[k]})" if unit[k] else ""),
                                       fmt, f"{v - r:+,.1f} vs 평균")
                    st.caption("참여 채널이 많고 반응 지연이 짧을수록 확산이 뜨겁다는 뜻입니다. "
                               "반대로 신규 영상은 계속 나오는데 영상당 댓글(반응효율)이 떨어지면 "
                               "관객이 먼저 떠나는 신호입니다.")

                    btype, bmsg = interpret_bass(bass_r2, qp)
                    st.markdown("**Bass 확산모형 진단**")
                    d1, d2 = st.columns([1, 3])
                    d1.metric("확산 유형", btype)
                    d2.info(f"{bmsg}\n\n— 모형 적합도 R²={bass_r2:.3f}, 모방/혁신 비율 q/p={qp:,.1f}")

                    with st.expander("주간 분석 데이터 보기"):
                        show = wk[["search", "rel", "n_comments", "n_channels_s",
                                   "lag_median_s", "eff_per_video_s", "n_new_videos_s",
                                   "gini_s", "weeks_from_peak"]].copy()
                        show.columns = ["검색량", "정점대비(%)", "댓글수", "참여채널",
                                        "반응지연(h)", "영상당댓글", "신규영상",
                                        "집중도", "정점기준주차"]
                        st.dataframe(show.round(2), use_container_width=True)

                    st.warning("이 예측은 14개 품목으로 학습한 **프로토타입**입니다. "
                               "품목 교차검증 AUC 0.716으로 검색량만 쓴 기준선(0.640)보다 낫지만, "
                               "실무 배포 기준(통상 0.8 이상)에는 미치지 못하며 품목별 편차가 큽니다. "
                               "발주 판단의 유일한 근거로 삼지 마세요.")

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
