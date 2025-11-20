from flask import Flask, jsonify, render_template, request
import os
import pandas as pd
import numpy as np
import re
from collections import Counter
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
DATA_PATH = os.path.join('data', 'GCASH_REVIEWS.csv')

# Custom Stopwords - Expanded to remove generic/useless words
STOPWORDS = set([
    'the', 'and', 'to', 'a', 'of', 'is', 'in', 'it', 'for', 'my', 'i', 'on', 'with', 'this', 
    'that', 'but', 'so', 'are', 'be', 'have', 'just', 'me', 'not', 'was', 'as', 'or', 'if', 
    'can', 'cant', 'cannot', 'do', 'dont', 'will', 'why', 'what', 'when', 'because', 'at',
    'na', 'ng', 'sa', 'ang', 'ko', 'mo', 'lang', 'po', 'pa', 'naman', 'yung', 'ung', 'ba', 
    'kasi', 'kaya', 'din', 'rin', 'may', 'mga', 'ni', 'kay', 'si', 'namin', 'ako', 'ka',
    'hindi', 'wag', 'nyo', 'niyo', 'sana', 'wala', 'dito', 'para',
    'app', 'gcash', 'application', 'please', 'pls', 'globe', 'account', 'number', 'use', 'time', 
    'update', 'phone', 'money', 'wallet',
    # Generic Adjectives / Adverbs (User requested removal)
    'good', 'bad', 'great', 'nice', 'best', 'better', 'worst', 'terrible', 'excellent', 'amazing',
    'very', 'really', 'too', 'so', 'much', 'many', 'lot', 'bit', 'little', 'more', 'less',
    'its', 'im', 'ive', 'dont', 'cant', 'wont', 'didnt', 'doesnt', 'isnt', 'arent', 'wasnt', 'werent',
    'youre', 'theyre', 'were', 'theres', 'heres', 'thats', 'whats', 'wheres', 'whos', 'hows',
    'gonna', 'wanna', 'gotta', 'lemme', 'gimme', 'kinda', 'sorta',
    'please', 'pls', 'plz', 'thanks', 'thank', 'thx', 'ty', 'salamat',
    'love', 'like', 'hate', 'dislike', 'enjoy', 'prefer',
    'make', 'made', 'doing', 'did', 'done', 'get', 'got', 'getting', 'go', 'going', 'gone',
    'know', 'knew', 'known', 'think', 'thought', 'feel', 'felt', 'see', 'saw', 'seen',
    'use', 'using', 'used', 'user', 'users', 'app', 'apps', 'application', 'applications',
    'gcash', 'globe', 'paymaya', 'coins', 'coinsph', 'palawan', 'cebuana', 'mlhuillier',
    'time', 'day', 'week', 'month', 'year', 'today', 'yesterday', 'tomorrow', 'now', 'then',
    'still', 'even', 'just', 'only', 'also', 'already', 'always', 'never', 'ever', 'sometimes',
    'back', 'front', 'side', 'top', 'bottom', 'left', 'right', 'center', 'middle',
    'way', 'ways', 'thing', 'things', 'something', 'anything', 'nothing', 'everything',
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'second', 'third', 'last', 'next', 'previous', 'past', 'future',
    'people', 'person', 'guy', 'girl', 'man', 'woman', 'kid', 'child', 'friend', 'family',
    'account', 'accounts', 'profile', 'profiles', 'number', 'numbers', 'mobile', 'mobiles'
])

# Business Categories for Smart Classification
CATEGORIES = {
    'Technical Stability': ['crash', 'lag', 'slow', 'bug', 'error', 'glitch', 'close', 'freeze', 'hang', 'open'],
    'Transactions': ['send', 'money', 'transfer', 'cash', 'load', 'refund', 'bank', 'gcredit', 'pay', 'bill'],
    'Security & Account': ['otp', 'login', 'verify', 'mpin', 'password', 'block', 'hack', 'id', 'verification', 'sim'],
    'Customer Service': ['support', 'agent', 'ticket', 'service', 'reply', 'chat', 'help', 'email', 'staff'],
    'User Experience': ['update', 'design', 'interface', 'easy', 'hard', 'confusing', 'feature', 'ad', 'screen']
}

try:
    print("⏳ Loading dataset...")
    df_global = pd.read_csv(DATA_PATH)
    
    # Normalize columns
    df_global.columns = [c.lower().strip() for c in df_global.columns]
    column_map = {
        'review_text': 'content', 'content': 'content',
        'review_rating': 'score', 'score': 'score',
        'review_datetime_utc': 'at', 'at': 'at',
        'review_likes': 'thumbsUpCount', 'thumbsupcount': 'thumbsUpCount',
        'author_name': 'userName', 'username': 'userName',
        'author_app_version': 'version', 'reviewcreatedversion': 'version'
    }
    df_global = df_global.rename(columns=column_map)
    
    # Convert Date
    if 'at' in df_global.columns:
        df_global['at'] = pd.to_datetime(df_global['at'], errors='coerce')
        df_global = df_global.dropna(subset=['at'])
        df_global['year'] = df_global['at'].dt.year
    
    # Convert Score
    if 'score' in df_global.columns:
        df_global['score'] = pd.to_numeric(df_global['score'], errors='coerce')
        
    # Fill NaNs
    df_global = df_global.where(pd.notnull(df_global), None)
    
    AVAILABLE_YEARS = sorted(df_global['year'].unique().astype(int).tolist(), reverse=True) if 'year' in df_global.columns else []

    print(f"✅ Data loaded: {len(df_global)} rows")

except Exception as e:
    print(f"❌ Error loading data: {e}")
    df_global = pd.DataFrame()
    AVAILABLE_YEARS = []

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_word_frequencies(text_series, top_n=20):
    if text_series.empty: return []
    all_text = " ".join(text_series.dropna().astype(str).tolist()).lower()
    all_text = re.sub(r'[^a-z\s]', '', all_text)
    words = all_text.split()
    clean_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return Counter(clean_words).most_common(top_n)

def categorize_review(text):
    """Classifies a review into a business category based on keywords."""
    text = str(text).lower()
    for cat, keywords in CATEGORIES.items():
        if any(k in text for k in keywords):
            return cat
    return 'General'

def generate_automated_insights(df_view, monthly_data):
    """Generates natural language insights based on the current data view."""
    insights = []
    
    if df_view.empty:
        return []

    # 1. RATING HEALTH
    avg_rating = df_view['score'].mean()
    if avg_rating < 3.0:
        insights.append({
            "type": "critical", 
            "title": "Low Customer Satisfaction",
            "text": f"Average rating is critical ({avg_rating:.2f}). Immediate investigation into recent updates is recommended."
        })
    elif avg_rating > 4.2:
        insights.append({
            "type": "success", 
            "title": "High User Sentiment",
            "text": f"Users are highly satisfied ({avg_rating:.2f}). Current strategy is effective."
        })

    # 2. TREND DIRECTION
    if len(monthly_data['ratings']) >= 2:
        last_month = monthly_data['ratings'][-1]
        prev_month = monthly_data['ratings'][-2]
        
        if last_month is not None and prev_month is not None:
            diff = last_month - prev_month
            if diff < -0.5:
                insights.append({
                    "type": "warning",
                    "title": "Negative Trend Detected",
                    "text": "Satisfaction has dropped significantly compared to the previous period."
                })
            elif diff > 0.5:
                insights.append({
                    "type": "success",
                    "title": "Positive Momentum",
                    "text": "Ratings are improving rapidly compared to the previous period."
                })
                
    return insights

# ==========================================
# 3. ROUTES
# ==========================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment')
def sentiment_page():
    return render_template('sentiment.html')

@app.route('/versions')
def versions_page():
    return render_template('versions.html')

@app.route('/reviews')
def reviews_page():
    return render_template('reviews.html')

# --- API: OVERVIEW (WITH PREDICTION & INSIGHTS) ---
@app.route('/api/overview')
def overview():
    try:
        if df_global.empty: return jsonify({'error': 'Dataset empty'}), 500
        
        start_date = request.args.get('start')
        end_date = request.args.get('end')
        target_year = request.args.get('urgent_year')

        df_view = df_global.copy()
        
        if start_date and end_date:
            mask = (df_view['at'] >= start_date) & (df_view['at'] <= end_date)
            df_view = df_view.loc[mask]

        # KPIs
        total_reviews = len(df_view)
        avg_rating = round(df_view['score'].mean(), 2) if total_reviews > 0 else 0
        
        if total_reviews > 0:
            pos = len(df_view[df_view['score'] >= 4])
            neg = len(df_view[df_view['score'] <= 2])
            net_sentiment = round(((pos - neg) / total_reviews) * 100, 1)
        else:
            net_sentiment = 0

        # Trends Chart (Actual Data)
        monthly_data = {'labels': [], 'ratings': []}
        if not df_view.empty:
            date_range_days = (df_view['at'].max() - df_view['at'].min()).days
            rule = 'D' if date_range_days < 60 else 'ME'
            fmt = '%Y-%m-%d' if rule == 'D' else '%b %Y'
            
            trends = df_view.set_index('at').resample(rule)['score'].mean()
            monthly_data = {
                'labels': [d.strftime(fmt) for d in trends.index],
                'ratings': [round(x, 2) if not pd.isna(x) else None for x in trends.tolist()] 
            }

        # Distribution Chart
        rating_dist = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
        if not df_view.empty:
            counts = df_view['score'].value_counts().to_dict()
            for k, v in counts.items():
                rating_dist[str(int(k))] = v

        # --- NEW: CATEGORY ANALYSIS ---
        category_data = {}
        if not df_view.empty:
            # Apply categorization to the current view
            # Note: For large datasets, this apply might be slow. 
            # In production, we would pre-compute this column on load.
            # For this project, we'll do it on the fly or check if we can optimize.
            # Optimization: Pre-compute on load if possible, but let's stick to on-fly for simplicity unless it's too slow.
            # Actually, let's pre-compute it on load to be safe.
            pass # We will handle this by pre-computing in the loading block if we can, or just here.
            
            # Let's do it here for now.
            df_view['category'] = df_view['content'].apply(categorize_review)
            cat_counts = df_view['category'].value_counts()
            
            labels = list(CATEGORIES.keys()) + ['General']
            data = [int(cat_counts.get(l, 0)) for l in labels]
            
            category_data = {
                'labels': labels,
                'data': data
            }
            
            # Add category insight
            top_cat = cat_counts.idxmax()
            top_cat_count = cat_counts.max()
            pct = (top_cat_count / total_reviews) * 100
            
            # Generate Insights
            insights = generate_automated_insights(df_view, monthly_data)
            if top_cat != 'General':
                insights.append({
                    "type": "info",
                    "title": f"Dominant Topic: {top_cat}",
                    "text": f"{pct:.1f}% of reviews relate to {top_cat}."
                })
        else:
            category_data = {'labels': [], 'data': []}
            insights = []

        # Urgent Issues
        df_urgent = df_global.copy()
        if target_year and target_year != 'All':
            df_urgent = df_urgent[df_urgent['year'] == int(target_year)]
            
        urgent_reviews = []
        if not df_urgent.empty:
            crit = df_urgent[df_urgent['score'] <= 2].sort_values('thumbsUpCount', ascending=False).head(5)
            for _, row in crit.iterrows():
                urgent_reviews.append({
                    'user': row.get('userName', 'Anonymous'),
                    'content': row.get('content', ''),
                    'score': row.get('score', 1),
                    'likes': int(row.get('thumbsUpCount', 0)),
                    'version': row.get('version', 'N/A'),
                    'date': row['at'].strftime('%Y-%m-%d')
                })

        # Prediction Logic
        prediction = None
        valid_indices = [i for i, v in enumerate(monthly_data['ratings']) if v is not None]
        valid_ratings = [monthly_data['ratings'][i] for i in valid_indices]
        
        if len(valid_ratings) >= 3: 
            try:
                X = np.array(valid_indices).reshape(-1, 1)
                y = np.array(valid_ratings)
                model = LinearRegression()
                model.fit(X, y)
                
                next_index = len(monthly_data['ratings'])
                pred_val = model.predict(np.array([[next_index]]))[0]
                pred_val = max(1.0, min(5.0, pred_val))
                
                trend_line = []
                full_range = list(range(len(monthly_data['ratings']) + 1))
                for i in full_range:
                    val = model.predict(np.array([[i]]))[0]
                    trend_line.append(round(max(1.0, min(5.0, val)), 2))

                last_val = valid_ratings[-1]
                prediction = {
                    'next_month_label': 'Forecast',
                    'predicted_rating': round(pred_val, 2),
                    'trend_direction': 'UP' if pred_val > last_val else 'DOWN',
                    'trend_line': trend_line
                }
            except Exception as e:
                print(f"Prediction error: {e}")
                prediction = None

        return jsonify({
            'total_reviews': total_reviews,
            'avg_rating': avg_rating,
            'net_sentiment': net_sentiment,
            'monthly_trends': monthly_data,
            'rating_distribution': rating_dist,
            'urgent_reviews': urgent_reviews,
            'available_years': AVAILABLE_YEARS,
            'prediction': prediction,
            'category_data': category_data,
            'insights': insights
        })

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

# --- API: SENTIMENT ---
@app.route('/api/sentiment')
def get_sentiment():
    try:
        if df_global.empty: return jsonify({'error': 'Dataset empty'}), 500
        
        start_date = request.args.get('start')
        end_date = request.args.get('end')
        
        df_view = df_global.copy()
        if start_date and end_date:
            mask = (df_view['at'] >= start_date) & (df_view['at'] <= end_date)
            df_view = df_view.loc[mask]

        if df_view.empty:
            return jsonify({'word_cloud': [], 'positive_words': [], 'negative_words': []})

        common_words = get_word_frequencies(df_view['content'], top_n=50)
        word_cloud_data = [{'text': w[0], 'weight': w[1]} for w in common_words]

        pos_df = df_view[df_view['score'] >= 4]
        pos_words = get_word_frequencies(pos_df['content'], top_n=10)
        
        neg_df = df_view[df_view['score'] <= 2]
        neg_words = get_word_frequencies(neg_df['content'], top_n=10)

        return jsonify({
            'word_cloud': word_cloud_data,
            'positive_words': {'labels': [x[0] for x in pos_words], 'data': [x[1] for x in pos_words]},
            'negative_words': {'labels': [x[0] for x in neg_words], 'data': [x[1] for x in neg_words]}
        })

    except Exception as e:
        print(f"Sentiment Error: {e}")
        return jsonify({'error': str(e)}), 500
    
# --- API: VERSIONS ---
@app.route('/api/versions')
def get_versions():
    try:
        if df_global.empty: return jsonify({'error': 'Dataset empty'}), 500
        
        stats = df_global.groupby('version').agg({'score': 'mean', 'content': 'count'}).reset_index()
        top_versions = stats.sort_values('content', ascending=False).head(12)
        leaderboard = top_versions.sort_values('score', ascending=False)
        
        rating_data = {
            'labels': leaderboard['version'].tolist(),
            'data': [round(x, 2) for x in leaderboard['score'].tolist()],
            'counts': leaderboard['content'].tolist()
        }

        tech_keywords = ['crash', 'lag', 'slow', 'bug', 'error', 'glitch', 'open', 'fix']
        crash_data = []

        for v in top_versions['version']:
            v_df = df_global[df_global['version'] == v]
            total = len(v_df)
            if total > 0:
                pattern = '|'.join(tech_keywords)
                issue_count = v_df['content'].astype(str).str.lower().str.contains(pattern, regex=True).sum()
                percentage = round((issue_count / total) * 100, 1)
                crash_data.append({
                    'version': v, 'percentage': percentage, 'count': int(issue_count), 'total': total
                })
        
        crash_data.sort(key=lambda x: x['percentage'], reverse=True)

        return jsonify({'rating_chart': rating_data, 'crash_data': crash_data})

    except Exception as e:
        print(f"Version Error: {e}")
        return jsonify({'error': str(e)}), 500

# --- API: REVIEWS ---
@app.route('/api/reviews')
def get_reviews():
    try:
        if df_global.empty: return jsonify({'error': 'Dataset empty'}), 500

        search_query = request.args.get('search', '').lower()
        filter_rating = request.args.get('rating')
        sort_by = request.args.get('sort', 'newest')
        limit = int(request.args.get('limit', 50))

        df_filtered = df_global.copy()

        if search_query:
            df_filtered = df_filtered[
                df_filtered['content'].astype(str).str.lower().str.contains(search_query) |
                df_filtered['userName'].astype(str).str.lower().str.contains(search_query)
            ]

        if filter_rating and filter_rating != 'all':
            df_filtered = df_filtered[df_filtered['score'] == int(filter_rating)]

        if sort_by == 'newest': df_filtered = df_filtered.sort_values('at', ascending=False)
        elif sort_by == 'oldest': df_filtered = df_filtered.sort_values('at', ascending=True)
        elif sort_by == 'highest': df_filtered = df_filtered.sort_values('score', ascending=False)
        elif sort_by == 'lowest': df_filtered = df_filtered.sort_values('score', ascending=True)

        df_filtered = df_filtered.head(limit)

        reviews_data = []
        for _, row in df_filtered.iterrows():
            reviews_data.append({
                'user': row.get('userName', 'Anonymous'),
                'content': row.get('content', ''),
                'score': int(row.get('score', 0)),
                'date': row['at'].strftime('%Y-%m-%d') if pd.notnull(row['at']) else 'N/A',
                'version': row.get('version', 'N/A'),
                'likes': int(row.get('thumbsUpCount', 0))
            })

        return jsonify({'count': len(reviews_data), 'reviews': reviews_data})

    except Exception as e:
        print(f"Search Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)