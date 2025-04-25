Python 3.12.5 (tags/v3.12.5:ff3bc82, Aug  6 2024, 20:45:27) [MSC v.1940 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # hybrid_recommender.py
... 
... import pandas as pd
... import numpy as np
... from difflib import get_close_matches
... from sklearn.feature_extraction.text import TfidfVectorizer
... from scipy.sparse import csr_matrix
... from sklearn.metrics.pairwise import cosine_similarity
... 
... # 1. Collaborative Filtering Function
... def get_cf_scores(data, user_id):
...     try:
...         user_item_matrix = data.pivot_table(index='ID', columns='Prod_ID', values='Rating')
...         if user_id not in user_item_matrix.index:
...             return pd.DataFrame(columns=['CF_Score'])
... 
...         user_similarity = user_item_matrix.corrwith(user_item_matrix.loc[user_id], axis=1, method='pearson')
...         sim_df = user_similarity.dropna().sort_values(ascending=False).drop(user_id, errors='ignore')
... 
...         similar_users = sim_df.head(5).index
...         similar_ratings = user_item_matrix.loc[similar_users]
... 
...         weighted_ratings = similar_ratings.T.dot(sim_df.loc[similar_users])
...         sim_sums = similar_ratings.notnull().T.dot(sim_df.loc[similar_users].abs())
... 
...         cf_scores = (weighted_ratings / sim_sums).dropna()
...         return pd.DataFrame(cf_scores, columns=['CF_Score'])
...     except Exception as e:
...         print(f"CF Error: {str(e)}")
...         return pd.DataFrame(columns=['CF_Score'])
... 
... # 2. Content-Based Filtering Function
... def get_cb_scores(data, item_name):
    try:
        matches = data[data['Name'].str.contains(item_name, case=False, na=False)]
        if len(matches) == 0:
            close_matches = get_close_matches(item_name, data['Name'].dropna().unique(), n=1, cutoff=0.6)
            if close_matches:
                matches = data[data['Name'] == close_matches[0]]
                print(f"Using closest match: {close_matches[0]}")

        if len(matches) == 0:
            return pd.DataFrame(columns=['Prod_ID', 'CB_Score'])

        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        desc_matrix = tfidf.fit_transform(data['Description'].fillna(''))
        cos_sim = cosine_similarity(desc_matrix[matches.index[0]], desc_matrix).flatten()

        return data[['Prod_ID']].assign(CB_Score=cos_sim).drop_duplicates()
    except Exception as e:
        print(f"CB Error: {str(e)}")
        return pd.DataFrame(columns=['Prod_ID', 'CB_Score'])

# 3. Hybrid Recommendation Function
def hybrid_recommendation(data, user_id, item_name, top_n=5, cf_weight=0.6, cb_weight=0.4):
    try:
        products = data[['Prod_ID', 'Name', 'Brand']].drop_duplicates()

        # Merge Collaborative Filtering Scores
        cf_scores = get_cf_scores(data, user_id)
        if not cf_scores.empty:
            products = products.merge(
                cf_scores.reset_index().rename(columns={'index': 'Prod_ID'}),
                on='Prod_ID',
                how='left'
            )

        # Merge Content-Based Scores
        cb_scores = get_cb_scores(data, item_name)
        if not cb_scores.empty:
            products = products.merge(cb_scores, on='Prod_ID', how='left')

        # Calculate Hybrid Score
        products = products.fillna(0)
        products['Hybrid_Score'] = (cf_weight * products['CF_Score']) + (cb_weight * products['CB_Score'])

        # Filter already rated items
        rated = set(data[data['ID'] == user_id]['Prod_ID'])
        return (
            products[~products['Prod_ID'].isin(rated)]
            .sort_values('Hybrid_Score', ascending=False)
            .head(top_n)
        )
    except Exception as e:
        print(f"Hybrid Error: {str(e)}")
        return pd.DataFrame(columns=['Prod_ID', 'Name', 'Brand', 'Hybrid_Score'])

# 4. Example usage (testing)
if __name__ == "__main__":
    # Sample data
    test_data = pd.DataFrame({
        'ID': [1, 1, 2, 2, 3, 3, 4, 4],
        'Prod_ID': [101, 102, 101, 103, 102, 103, 101, 102],
        'Name': ['Lipstick A', 'Lipstick B', 'Lipstick A', 'Lipstick C',
                 'Lipstick B', 'Lipstick C', 'Lipstick A', 'Lipstick B'],
        'Brand': ['Brand1', 'Brand2', 'Brand1', 'Brand3',
                  'Brand2', 'Brand3', 'Brand1', 'Brand2'],
        'Rating': [4, 3, 5, 2, 4, 1, 3, 5],
        'Description': ['matte red lipstick', 'glossy pink lipstick',
                        'matte red lipstick', 'shiny nude lipstick',
                        'glossy pink lipstick', 'shiny nude lipstick',
                        'matte red lipstick', 'glossy pink lipstick']
    })

    # Test parameters
    target_user_id = 4
    item_name = 'Lipstick A'

    # Generate and print recommendations
    recommendations = hybrid_recommendation(test_data, target_user_id, item_name, top_n=5)
    if not recommendations.empty:
        print(f"\nTop Recommendations for user {target_user_id} based on '{item_name}':")
        print(recommendations.to_string(index=False))
    else:
        print("No recommendations could be generated.")
