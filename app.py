# ===== app.py : 유튜브 댓글 + 구글 검색량 추이 + 합친 파일 =====
import time
import requests
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

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

# ---------- (2) 구글 검색량 추이 수집 (SerpApi) ----------
def get_search_trend(keyword, start, end, serpapi_key):
    if not serpapi_key:
        st.info("검색량 추이는 SerpApi 키가 있어야 나와요. (사이드바에 입력)")
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

# ---------- (3) 두 파일 합치기 (댓글은 그대로, 그 달 검색량을 붙임) ----------
def merge_files(df_comments, df_trend):
    if df_trend.empty:
        out = df_comments.copy()
        out["search_interest"] = pd.NA
        return out
    t = df_trend.copy()
    t["comment_year_month"] = pd.to_datetime(t["date"]).dt.to_period("M").astype(str)
    s_month = t.groupby("comment_year_month")["search_interest"].mean().round(1).reset_index()
    return df_comments.merge(s_month, on="comment_year_month", how="left")

# ---------- 화면 ----------
st.title("🔎 유행 데이터 수집기")
st.caption("검색어를 넣으면 유튜브 댓글·구글 검색량 추이·둘을 합친 파일을 만듭니다.")

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
    max_videos = st.slider("영상 수", 5, 50, 30)
    max_comments = st.slider("영상당 댓글 수", 20, 200, 100, step=10)

col1, col2 = st.columns([4, 1])
keyword = col1.text_input("검색어", placeholder="예: 탕후루", label_visibility="collapsed")
run = col2.button("검색", type="primary", use_container_width=True)

if run:
    if not yt_key:
        st.error("YouTube API 키를 입력하세요. (사이드바)")
    elif not keyword.strip():
        st.warning("검색어를 입력하세요.")
    else:
        kw = keyword.strip()
        with st.spinner(f"'{kw}' 유튜브 댓글 수집 중..."):
            df = crawl_comments(kw, yt_key, max_videos, max_comments)
        if df.empty:
            st.error("수집된 댓글이 없어요. 검색어나 API 키를 확인해 주세요.")
        else:
            start = df["comment_published_at"].min().strftime("%Y-%m-%d")
            end   = df["comment_published_at"].max().strftime("%Y-%m-%d")
            with st.spinner("구글 검색량 추이 수집 중..."):
                trend = get_search_trend(kw, start, end, serp_key)
            merged = merge_files(df, trend)

            c1, c2, c3 = st.columns(3)
            c1.metric("수집 댓글", f"{len(df):,}")
            c2.metric("영상 수", f"{df['video_id'].nunique():,}")
            c3.metric("댓글 기간", f"{df['comment_published_at'].min().date()} ~ {df['comment_published_at'].max().date()}")

            st.subheader("① 유튜브 댓글")
            st.dataframe(df, use_container_width=True, height=260)
            st.download_button("댓글 CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{kw}_comments.csv", mime="text/csv")

            st.subheader("② 구글 검색량 추이 (0~100)")
            if trend.empty:
                st.info("검색량 추이를 못 가져왔어요. SerpApi 키/사용량을 확인해 주세요.")
            else:
                st.line_chart(trend.set_index("date")["search_interest"])
                st.download_button("검색량 CSV", trend.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"{kw}_search_trend.csv", mime="text/csv")

            st.subheader("③ 합친 파일 (댓글 + 그 달 검색량)")
            st.dataframe(merged, use_container_width=True, height=260)
            st.download_button("⭐ 합친 CSV 다운로드",
                               merged.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{kw}_merged.csv", mime="text/csv")
else:
    st.info("검색어를 넣고 '검색'을 누르세요.")
