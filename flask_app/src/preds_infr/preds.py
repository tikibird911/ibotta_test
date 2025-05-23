from flask import jsonify

def get_top_features(model, feature_names, top_n=5):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1][:top_n]
    return [
        {"feature": feature_names[i], "importance": float(importances[i])}
        for i in indices
    ]

def predict_by_customer_id(latest_model, latest_journey_df, customer_id):
    if latest_model is None or latest_journey_df is None:
        return jsonify({"error": "No model or data loaded yet."})
    if not customer_id:
        return jsonify({"error": "No customer_id provided."})
    print("customer_id received:", customer_id)
    print("latest_journey_df head:")
    print(latest_journey_df.head())
    df = latest_journey_df[latest_journey_df["customer_id"].astype(str) == str(customer_id)]
    if df.empty:
        return jsonify({"error": "No data found for this customer_id."})

    feature_cols = [
    'total_price', 'total_quantity', 'avg_price', 'median_price', 'std_price',
    'n_unique_categories', 'n_steps', 'most_common_category_encoded',
    'journey_degree_centrality_mean', 'journey_degree_centrality_max', 'journey_degree_centrality_median', 'journey_degree_centrality_std',
    'journey_betweenness_centrality_mean', 'journey_betweenness_centrality_max', 'journey_betweenness_centrality_median', 'journey_betweenness_centrality_std',
    'journey_closeness_centrality_mean', 'journey_closeness_centrality_max', 'journey_closeness_centrality_median', 'journey_closeness_centrality_std']
    X = df[feature_cols]
    preds = latest_model.predict(X)
    df = df.copy()
    df['prediction'] = preds

    # Get top 5 important features
    top_features = get_top_features(latest_model, feature_cols, top_n=5)

    # Calculate averages for the top features
    top_feature_names = [f['feature'] for f in top_features]
    averages = {feat: df[feat].mean() for feat in top_feature_names}

    return jsonify({
        "predictions": df[['customer_id', 'receipt_id', 'prediction']].to_dict(orient='records'),
        "top_features": top_features,
        "top_feature_averages": averages
    })