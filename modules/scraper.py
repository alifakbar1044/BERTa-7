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
                r['app_id'] = app_id
            all_reviews.extend(rvs)
        except Exception as e:
            print(f"Gagal mengambil {app_id}: {e}")
            
    df = pd.DataFrame(all_reviews)
    expected_cols = ['content', 'score', 'at', 'app_id']
    
    if df.empty:
        return pd.DataFrame(columns=expected_cols)
    available_cols = [c for c in expected_cols if c in df.columns]
    df = df[available_cols]
    
    return df