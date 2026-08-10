# ===== app.py : 검색창이 있는 유튜브 댓글 크롤링 웹앱 =====
import time
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

st.set_page_config(page_title="유튜브 댓글 크롤러", page_icon="🔎", layout="wide")

# ---------- 크롤러 (당신 파트) ----------
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
    prog = st.progress(0.0, text="댓글 수집 중...")
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
        prog.progress(i / max(1, len(ids)), text=f"댓글 수집 중... ({i}/{len(ids)})")
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

# ---------- 화면 ----------
st.title("🔎 유튜브 댓글 크롤러")
st.caption("검색어를 넣으면 관련 유튜브 영상의 댓글을 수집합니다.")

# API 키: 배포 시엔 st.secrets, 로컬/테스트 땐 입력창
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

# 네이버처럼 검색창 + 버튼
col1, col2 = st.columns([4, 1])
keyword = col1.text_input("검색어", placeholder="예: 탕후루", label_visibility="collapsed")
run = col2.button("검색", type="primary", use_container_width=True)

if run:
    if not api_key:
        st.error("API 키를 입력하세요. (사이드바)")
    elif not keyword.strip():
        st.warning("검색어를 입력하세요.")
    else:
        with st.spinner(f"'{keyword}' 댓글 수집 중..."):
            df = crawl_comments(keyword.strip(), api_key, max_videos, max_comments)
        if df.empty:
            st.error("수집된 댓글이 없어요. 검색어나 API 키를 확인해 주세요.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("수집 댓글", f"{len(df):,}")
            c2.metric("영상 수", f"{df['video_id'].nunique():,}")
            c3.metric("기간", f"{df['comment_published_at'].min().date()} ~ {df['comment_published_at'].max().date()}")
            st.dataframe(df, use_container_width=True, height=400)
            st.download_button("CSV 다운로드", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"{keyword}_comments.csv", mime="text/csv")
else:
    st.info("검색어를 넣고 '검색'을 누르세요.")
