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
    t["comment_year_month"] = pd.to_datetime(t["date"]).dt.to_period("
