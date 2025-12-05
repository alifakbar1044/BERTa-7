from google_play_scraper import Sort, reviews
import pandas as pd

def scrape_google_play(app_ids, count=200):
    all_reviews = []
    
    for app_id in app_ids:
        try:
            rvs, _ = reviews(
                app_id,
                lang='id',
                country='id',
                sort=Sort.NEWEST,
                count=count
            )
            for r in rvs:
                r['source_app'] = app_id
            all_reviews.extend(rvs)
        except Exception as e:
            print(f"Gagal mengambil {app_id}: {e}")
            
    df = pd.DataFrame(all_reviews)
    if not df.empty:
        df = df[['userName', 'content', 'score', 'at', 'source_app']]
    return df