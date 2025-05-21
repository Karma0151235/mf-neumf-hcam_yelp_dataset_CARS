import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def load_yelp(path_json_dir, sample_size=50000):
    business_path = os.path.join(path_json_dir, 'yelp_academic_dataset_business.json')
    review_path = os.path.join(path_json_dir, 'yelp_academic_dataset_review.json')

    business_df = pd.read_json(business_path, lines=True)
    business_df = business_df[business_df['categories'].str.contains('Restaurants', na=False)]
    restaurant_ids = set(business_df['business_id'])

    sample_reviews = []
    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record['business_id'] in restaurant_ids and record['date'] >= '2018-01-01':
                sample_reviews.append(record)
            if len(sample_reviews) >= sample_size:
                break

    reviews_df = pd.DataFrame(sample_reviews)

    if reviews_df.empty:
        raise ValueError("Sampled reviews are empty. Try increasing sample_size or checking your filters.")

    df = reviews_df.merge(business_df[['business_id', 'city', 'stars', 'review_count']], on='business_id')
    df = df[['user_id', 'business_id', 'stars_x', 'date', 'city', 'stars_y', 'review_count']]
    df.columns = ['user', 'item', 'rating', 'timestamp', 'city', 'biz_stars', 'review_count']
    return df

def preprocess(df, min_uc=3, min_ic=3):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df[df.groupby('user')['item'].transform('count') >= min_uc]
    df = df[df.groupby('item')['user'].transform('count') >= min_ic]

    if len(df) == 0:
        raise ValueError("Filtered dataset is empty. Try reducing min_uc/min_ic or increasing sample_size.")

    df = df.sort_values('timestamp').reset_index(drop=True)

    if df[['biz_stars', 'review_count']].dropna().shape[0] == 0:
        raise ValueError("No valid rows for scaling. Check input data or filtering thresholds.")

    scaler = MinMaxScaler()
    df[['biz_stars', 'review_count']] = scaler.fit_transform(df[['biz_stars', 'review_count']])

    top_cities = df['city'].value_counts().nlargest(50).index
    df['city'] = df['city'].where(df['city'].isin(top_cities), other='Other')
    city_ohe = pd.get_dummies(df['city'], prefix='city')
    df = pd.concat([df, city_ohe], axis=1)

    ctx = df[['biz_stars', 'review_count'] + list(city_ohe.columns)].values

    user2idx = {u: i for i, u in enumerate(df['user'].unique())}
    item2idx = {i: j for j, i in enumerate(df['item'].unique())}
    df['user'] = df['user'].map(user2idx)
    df['item'] = df['item'].map(item2idx)

    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/yelp_filtered.parquet")
    return df[['user', 'item', 'rating', 'timestamp']], ctx