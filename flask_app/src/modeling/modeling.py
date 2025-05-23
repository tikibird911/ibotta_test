import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import classification_report


def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Remove rows with missing values
    df = df.dropna()

    # Indicate buy/return: positive price = buy, negative price = return
    df['transaction_type'] = df['price'].apply(lambda x: 'buy' if x > 0 else 'return')

    # Remove rows where price is zero
    df = df[df['price'] != 0]

    return df


def build_customer_journey_graph_with_engager(df):
    G = nx.Graph()  # Use an undirected graph
    customer_status = {}

    for customer_id, customer_group in df.groupby('customer_id'):
        engager_any_yes = False
        for receipt_id, group in customer_group.groupby('receipt_id'):
            journey = group.sort_values('primary_category_name')  # Sort by category just for consistency
            node_ids = []
            for idx, row in journey.iterrows():
                primary_category = row['primary_category_name'] if pd.notna(row['primary_category_name']) else "Unknown"
                if primary_category in ["No Info", "Unknown", ""]:
                    continue
                secondary_category = row['secondary_category'] if pd.notna(row.get('secondary_category', None)) else None
                brand_name = row['brand_name'] if pd.notna(row.get('brand_name', None)) else None

                node_id = f"{customer_id}_{receipt_id}_{primary_category}"
                G.add_node(
                    node_id,
                    primary_category_name=primary_category,
                    price=row['price'],
                    quantity=row['quantity'],
                    secondary_category=secondary_category,
                    brand_name=brand_name,
                    customer_id=customer_id,
                    receipt_id=receipt_id,
                    transaction_type=row['transaction_type'],
                )
                node_ids.append(node_id)
            # Add edges between all pairs of items (primary_category_name) in the same receipt, with edge weights as counts
            for i in range(len(node_ids)):
                for j in range(i + 1, len(node_ids)):
                    u, v = node_ids[i], node_ids[j]
                    if G.has_edge(u, v):
                        G[u][v]['weight'] += 1
                    else:
                        G.add_edge(u, v, weight=1)
            # Add final engager node as "engager_yes" or "engager_no" (general, not customer/receipt specific)
            if 'is_engager' in journey.columns:
                engager_value = journey.iloc[-1]['is_engager']
                if pd.isna(engager_value):
                    engager_value = "no"
                else:
                    engager_value = str(engager_value).strip().lower()
                if engager_value == "yes":
                    engager_any_yes = True
                    engager_node = "engager_yes"
                    G.add_node(engager_node, engager="yes")
                else:
                    engager_node = "engager_no"
                    G.add_node(engager_node, engager="no")
                # Connect all nodes in this receipt to the engager node, with edge weight as count
                for node_id in node_ids:
                    if G.has_edge(node_id, engager_node):
                        G[node_id][engager_node]['weight'] += 1
                    else:
                        G.add_edge(node_id, engager_node, weight=1)
        customer_status[customer_id] = "yes" if engager_any_yes else "no"

    for n, data in G.nodes(data=True):
        cust_id = data.get('customer_id')
        if cust_id in customer_status:
            G.nodes[n]['customer_engager_status'] = customer_status[cust_id]

    return G, customer_status


def nx_to_igraph(G):
    # Exclude engager nodes
    nodes = [n for n in G.nodes if not str(n).startswith("engager_")]
    mapping = {n: i for i, n in enumerate(nodes)}
    # Only take u and v for undirected graph, and only if both are not engager nodes
    edges = [(mapping[u], mapping[v]) for u, v in G.edges if u in mapping and v in mapping]
    weights = [G[u][v].get('weight', 1) for u, v in G.edges if u in mapping and v in mapping]
    g = ig.Graph(edges=edges, directed=False)
    g.es['weight'] = weights
    return g, mapping


def igraph_closeness_centrality(G):
    g, mapping = nx_to_igraph(G)
    # Use weights as distances (inverse of weight for "distance")
    # If higher weight means closer, use 1/weight as distance
    distances = [1/w if w > 0 else 1 for w in g.es['weight']]
    closeness = g.closeness(weights=distances)
    reverse_mapping = {v: k for k, v in mapping.items()}
    return {reverse_mapping[i]: c for i, c in enumerate(closeness)}

def add_centrality_metrics(G, betweenness_k=500):
    nodes = [n for n in G.nodes if not str(n).startswith("engager_")]
    subG = G.subgraph(nodes)
    degree_centrality = nx.degree_centrality(subG)
    # Use weights for betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(subG, k=min(betweenness_k, len(subG)), weight='weight')
    closeness_centrality = igraph_closeness_centrality(subG)

    for n in subG.nodes:
        G.nodes[n]['degree_centrality'] = degree_centrality.get(n, 0)
        G.nodes[n]['betweenness_centrality'] = betweenness_centrality.get(n, 0)
        G.nodes[n]['closeness_centrality'] = closeness_centrality.get(n, 0)
    for n in G.nodes:
        if str(n).startswith("engager_"):
            G.nodes[n]['degree_centrality'] = 0
            G.nodes[n]['betweenness_centrality'] = 0
            G.nodes[n]['closeness_centrality'] = 0


def create_journey_df_with_last_node_centrality(G, df):
    """
    For each journey, find the last node and extract its centrality metrics,
    along with basic journey features. Returns a new journey_df.
    """
    features = []
    for (customer_id, receipt_id), group in df.groupby(['customer_id', 'receipt_id']):
        last_row = group.iloc[-1]
        primary_category = last_row['primary_category_name'] if pd.notna(last_row['primary_category_name']) else "Unknown"
        if primary_category in ["No Info", "Unknown", ""]:
            continue
        node_id = f"{customer_id}_{receipt_id}_{primary_category}_{last_row['created_at']}"
        node_data = G.nodes.get(node_id, {})
        is_engager = str(last_row['is_engager']).strip().lower() == "yes"
        # Median and std for buys and returns
        buys = group[group['transaction_type'] == 'buy']['price']
        returns = group[group['transaction_type'] == 'return']['price']
        median_buys = buys.median() if not buys.empty else 0
        std_buys = buys.std() if len(buys) > 1 else 0
        median_returns = returns.median() if not returns.empty else 0
        std_returns = returns.std() if len(returns) > 1 else 0
        features.append({
            'customer_id': customer_id,
            'receipt_id': receipt_id,
            'total_price': group['price'].sum(),
            'total_quantity': group['quantity'].sum(),
            'avg_price': group['price'].mean(),
            'avg_quantity': group['quantity'].mean(),
            'median_price': group['price'].median(),
            'median_quantity': group['quantity'].median(),
            'std_price': group['price'].std(),
            'std_quantity': group['quantity'].std(),
            'n_unique_categories': group['primary_category_name'].nunique(),
            'n_steps': len(group),
            'most_common_category': group['primary_category_name'].mode().iloc[0] if not group['primary_category_name'].mode().empty else "Unknown",
            'is_engager': int(is_engager),
            'n_buys': (group['transaction_type'] == 'buy').sum(),
            'n_returns': (group['transaction_type'] == 'return').sum(),
            'prop_buys': (group['transaction_type'] == 'buy').sum() / len(group) if len(group) > 0 else 0,
            'prop_returns': (group['transaction_type'] == 'return').sum() / len(group) if len(group) > 0 else 0,
            'median_buys': median_buys,
            'std_buys': std_buys,
            'median_returns': median_returns,
            'std_returns': std_returns,
            'degree_centrality': node_data.get('degree_centrality', 0),
            'betweenness_centrality': node_data.get('betweenness_centrality', 0),
            'closeness_centrality': node_data.get('closeness_centrality', 0)
        })
    journey_df = pd.DataFrame(features)
    return journey_df


def journey_centrality_features(G, df):
    features = []
    for (customer_id, receipt_id), group in df.groupby(['customer_id', 'receipt_id']):
        # No need to sort by 'created_at' for undirected graph and new node_id
        node_ids = []
        for idx, row in group.iterrows():
            primary_category = row['primary_category_name'] if pd.notna(row['primary_category_name']) else "Unknown"
            if primary_category in ["No Info", "Unknown", ""]:
                continue
            node_id = f"{customer_id}_{receipt_id}_{primary_category}"
            node_ids.append(node_id)
        # Get centrality values for all nodes in the journey
        deg = [G.nodes[n].get('degree_centrality', 0) for n in node_ids if n in G.nodes]
        bet = [G.nodes[n].get('betweenness_centrality', 0) for n in node_ids if n in G.nodes]
        clo = [G.nodes[n].get('closeness_centrality', 0) for n in node_ids if n in G.nodes]
        features.append({
            'customer_id': customer_id,
            'receipt_id': receipt_id,
            'journey_degree_centrality_mean': np.mean(deg) if deg else 0,
            'journey_degree_centrality_max': np.max(deg) if deg else 0,
            'journey_degree_centrality_median': np.median(deg) if deg else 0,
            'journey_degree_centrality_std': np.std(deg) if deg else 0,
            'journey_betweenness_centrality_mean': np.mean(bet) if bet else 0,
            'journey_betweenness_centrality_max': np.max(bet) if bet else 0,
            'journey_betweenness_centrality_median': np.median(bet) if bet else 0,
            'journey_betweenness_centrality_std': np.std(bet) if bet else 0,
            'journey_closeness_centrality_mean': np.mean(clo) if clo else 0,
            'journey_closeness_centrality_max': np.max(clo) if clo else 0,
            'journey_closeness_centrality_median': np.median(clo) if clo else 0,
            'journey_closeness_centrality_std': np.std(clo) if clo else 0,
        })
    return pd.DataFrame(features)


def train_xgb_model(journey_df, show_importance_table=True):
    """
    Trains an XGBoost classifier on journey_df and returns the trained model,
    the classification report (as a string), and the Plotly figure (if plot_importance=True).
    """
    # Encode most_common_category as integer
    le = LabelEncoder()
    journey_df['most_common_category_encoded'] = le.fit_transform(journey_df['most_common_category'])

    feature_cols = [
        'total_price', 'total_quantity', 'avg_price',
        'median_price', 'std_price',
        'n_unique_categories', 'n_steps', 'most_common_category_encoded',
        'journey_degree_centrality_mean', 'journey_degree_centrality_max', 'journey_degree_centrality_median', 'journey_degree_centrality_std',
        'journey_betweenness_centrality_mean', 'journey_betweenness_centrality_max', 'journey_betweenness_centrality_median', 'journey_betweenness_centrality_std',
        'journey_closeness_centrality_mean', 'journey_closeness_centrality_max', 'journey_closeness_centrality_median', 'journey_closeness_centrality_std',
        # 'centrality_cluster'
    ]

    X = journey_df[feature_cols]
    y = journey_df['is_engager']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train XGBoost model
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    importance_table_html = None
    if show_importance_table:
        importances = model.feature_importances_
        indices = importances.argsort()[::-1][:15]  # Top 15 features
        top_features = [feature_cols[i] for i in indices]
        top_importances = importances[indices]

        # Option 1: Return as HTML table
        rows = [
            f"<tr><td>{feat}</td><td>{imp:.4f}</td></tr>"
            for feat, imp in zip(top_features, top_importances)
        ]
        importance_table_html = (
            "<table border='1'><tr><th>Feature</th><th>Importance</th></tr>"
            + "".join(rows) +
            "</table>"
        )

    return model, report, importance_table_html


def model_QB(excel_file):
    df = pd.read_excel(excel_file)
    df = clean_data(df)
    G, customer_status = build_customer_journey_graph_with_engager(df)  
    add_centrality_metrics(G)
    journey_df = create_journey_df_with_last_node_centrality(G, df)
    centrality_df = journey_centrality_features(G, df)
    # Merge on customer_id and receipt_id
    journey_df = pd.merge(journey_df, centrality_df, on=['customer_id', 'receipt_id'], how='left')
    model, report, importance_table_html = train_xgb_model(journey_df)

    return model, report, importance_table_html, journey_df