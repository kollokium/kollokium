# ===== app.py : 유튜브 댓글 + 구글 검색량 추이(일별) 수집 웹앱 =====
import time
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pytrends.request import TrendReq

st.set_page_config(page_title="유행 데이터 수집기", page_icon="🔎", layout="wide")

# ---------- (1) 유튜브 댓글 크롤러 ----------
def crawl_comments(keyword, api_key, max_videos=30, max_comments=100):
    yt = build("youtube", "v3", developerKey=api_key)
    ids, seen, page = [], set(), None
    while len(ids) < max_videos:
        res = yt.search().list(q=keyword, part="id", type="video",
            maxResults=min(50, max_videos - len(ids)), pageToken=page,
            regionCode="KR", relevanceLanguage="ko").execute()
        for it in res.get("items", []):
            v = it["id"]["videoId"]
            if v not in seen:
                seen.add(v); ids.append(v)
        page = res.get("nextPageToken")
        if not page:
            break
    details = {}
    for i in range(0, len(ids), 50):
        res = yt.videos().list(part="snippet,statistics",
                               id=",".join(ids[i:i+50])).execute()
        for it in res.get("items", []):
            s, stt = it["snippet"], it.get("statistics", {})
            details[it["id"]] = {
                "video_title": s["title"], "channel_title": s["channelTitle"],
                "view_count": int(stt["viewCount"]) if "viewCount" in stt else None,
                "video_published_at": s["publishedAt"]}
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
    df["comment_year_month"] = df["comment_published_at"].dt.tz_convert(None).dt.to_period("M").astype(str)
    df.sort_values("comment_published_at", inplace=True)
    return df.reset_index(drop=True)

# ---------- (2) 구글 검색량 추이 수집 (일별, 429 재시도 포함) ----------
def get_search_trend(keyword, start, end, retries=3):
    """댓글 기간에 맞춰 일별 검색 관심도(0~100) 수집. 429면 쉬었다 재시도."""
    for attempt in range(retries):
        try:
            pytrends = TrendReq(hl="ko", tz=540, timeout=(10, 25),
                                retries=2, backoff_factor=0.5)
            pytrends.build_payload([keyword], geo="KR", timeframe=f"{start} {end}")
            t = pytrends.interest_over_time()
            if t.empty:
                return pd.DataFrame()
            t = t.reset_index()[["date", keyword]]
            t.columns = ["date", "search_interest"]
            t["date"] = pd.to_datetime(t["date"]).dt.date   # 일별 그대로
            return t
        except Exception:
            if attempt < retries - 1:
                time.sleep(10)      # 10초 쉬고 재시도
                continue
            st.warning("검색량 추이 수집 실패(구글 호출 제한 429). 잠시 후 다시 검색해 보세요.")
            return pd.DataFrame()

# ---------- 화면 ----------
st.title("🔎 유행 데이터 수집기")
st.caption("검색어를 넣으면 유튜브 댓글과 구글 검색량 추이(일별)를 함께 수집합니다.")

api_key = ""
try:
    api_key = st.secrets["YOUTUBE_API_KEY"]
except Exception:
    api_key = ""

with st.sidebar:
    st.header("설정")
    if not api_key:
        api_key = st.text_input("YouTube API 키", type="password")
    max_videos = st.slider("영상 수", 5, 50, 30)
    max_comments = st.slider("영상당 댓글 수", 20, 200, 100, step=10)

col1, col2 = st.columns([4, 1])
keyword = col1.text_input("검색어", placeholder="예: 탕후루", label_visibility="collapsed")
run = col2.button("검색", type="primary", use_container_width=True)

if run:
    if not api_key:
        st.error("API 키를 입력하세요. (사이드바)")
    elif not keyword.strip():
        st.warning("검색어를 입력하세요.")
    else:
        kw = keyword.strip()
        with st.spinner(f"'{kw}' 유튜브 댓글 수집 중..."):
            df = crawl_comments(kw, api_key, max_videos, max_comments)
        if df.empty:
            st.error("수집된 댓글이 없어요. 검색어나 API 키를 확인해 주세요.")
        else:
            start = df["comment_published_at"].min().strftime("%Y-%m-%d")
            end   = df["comment_published_at"].max().strftime("%Y-%m-%d")
            with st.spinner("구글 검색량 추이 수집 중... (최대 30초)"):
                trend = get_search_trend(kw, start, end)

            c1, c2, c3 = st.columns(3)
            c1.metric("수집 댓글", f"{len(df):,}")
            c2.metric("영상 수", f"{df['video_id'].nunique():,}")
            c3.metric("댓글 기간", f"{df['comment_published_at'].min().date()} ~ {df['comment_published_at'].max().date()}")

            st.subheader("① 유튜브 댓글")
            st.dataframe(df, use_container_width=True, height=320)
            st.download_button("댓글 CSV 다운로드", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{kw}_comments.csv", mime="text/csv")

            st.subheader("② 구글 검색량 추이 (일별, 0~100)")
            if trend.empty:
                st.info("검색량 추이를 못 가져왔어요. 구글 호출 제한(429)일 수 있으니 잠시 후 다시 검색해 보세요.")
            else:
                st.line_chart(trend.set_index("date")["search_interest"])
                st.dataframe(trend, use_container_width=True)
                st.download_button("검색량 CSV 다운로드", trend.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"{kw}_search_trend.csv", mime="text/csv")
else:
    st.info("검색어를 넣고 '검색'을 누르세요.")
