from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from typing import List, Optional

# Example of the hybrid recommendation function
def hybrid_recommendation(data, user_id, item_name, top_n=10, cf_weight=0.5, cb_weight=0.5):
    """
    Hybrid recommendation combining collaborative and content-based filtering.
    Returns DataFrame with: Prod_ID, Name, Brand, Hybrid_Score, CF_Score, CB_Score
    """
    def get_cf_scores(data, user_id):
        try:
            user_item = data.pivot_table(index='ID', columns='Prod_ID', values='Rating', aggfunc='mean').fillna(0)
            sparse_mat = csr_matrix(user_item.values)
            user_sim = cosine_similarity(sparse_mat)

            user_pos = user_item.index.get_loc(user_id)
            sim_users = [(idx, score) for idx, score in enumerate(user_sim[user_pos])
                         if idx != user_pos and score > 0.3]
            sim_users.sort(key=lambda x: x[1], reverse=True)

            rated_items = set(data[data['ID'] == user_id]['Prod_ID'])
            item_scores = {}
            for item in set(user_item.columns) - rated_items:
                total = sum(user_item.iloc[user_idx][item] * sim_score
                            for user_idx, sim_score in sim_users[:20]
                            if user_item.iloc[user_idx][item] > 0)
                sim_sum = sum(sim_score for user_idx, sim_score in sim_users[:20]
                              if user_item.iloc[user_idx][item] > 0)
                if sim_sum > 0:
                    item_scores[item] = total / sim_sum

            return pd.DataFrame.from_dict(item_scores, orient='index', columns=['CF_Score'])
        except Exception as e:
            return pd.DataFrame(columns=['CF_Score'])

    def get_cb_scores(data, item_name):
        try:
            matches = data[data['Name'].str.contains(item_name, case=False, na=False)]
            if len(matches) == 0:
                return pd.DataFrame(columns=['Prod_ID', 'CB_Score'])
            tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
            desc_matrix = tfidf.fit_transform(data['Description'].fillna(''))
            cos_sim = cosine_similarity(desc_matrix[matches.index[0]], desc_matrix).flatten()
            return data[['Prod_ID']].assign(CB_Score=cos_sim).drop_duplicates()
        except Exception as e:
            return pd.DataFrame(columns=['Prod_ID', 'CB_Score'])

    try:
        products = data[['Prod_ID', 'Name', 'Brand']].drop_duplicates()

        cf_scores = get_cf_scores(data, user_id)
        if not cf_scores.empty:
            products = products.merge(cf_scores.reset_index().rename(columns={'index': 'Prod_ID'}),
                                      on='Prod_ID', how='left')

        cb_scores = get_cb_scores(data, item_name)
        if not cb_scores.empty:
            products = products.merge(cb_scores, on='Prod_ID', how='left')
        else:
            products['CB_Score'] = 0

        products = products.fillna(0)
        products['Hybrid_Score'] = (cf_weight * products['CF_Score']) + (cb_weight * products['CB_Score'])

        rated = set(data[data['ID'] == user_id]['Prod_ID'])
        return (products[~products['Prod_ID'].isin(rated)]
                .sort_values('Hybrid_Score', ascending=False)
                .head(top_n))

    except Exception as e:
        return pd.DataFrame(columns=['Prod_ID', 'Name', 'Brand', 'Hybrid_Score'])

# Define FastAPI app
app = FastAPI()

# Define the input and output models
class RecommendationRequest(BaseModel):
    user_id: int
    item_name: str
    top_n: Optional[int] = 10

class RecommendationResponse(BaseModel):
    Prod_ID: int
    Name: str
    Brand: str
    Hybrid_Score: float
    CF_Score: Optional[float] = 0
    CB_Score: Optional[float] = 0

# Example of loading your data (this should be replaced with your actual data)
train_data = pd.read_csv('hybrid_recommendations.csv')

@app.post("/recommendations", response_model=List[RecommendationResponse])
async def get_hybrid_recommendations(request: RecommendationRequest):
    try:
        recommendations = hybrid_recommendation(
            train_data,
            user_id=request.user_id,
            item_name=request.item_name,
            top_n=request.top_n
        )

        # Convert the recommendations to the response model
        recommendations_list = recommendations.to_dict(orient='records')
        return recommendations_list
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error generating recommendations")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
