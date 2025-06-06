import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from transformers import pipeline

# --- Page config ---
st.set_page_config(page_title="E-Commerce Returns", layout='wide')

# --- Load sentiment model ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device=-1)

sentiment_analyzer = load_sentiment_model()


# --- CSS ---
css = """
<style>
body {
    background-color: #F5F5F5;
    font-family: 'Arial', sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #003366;
    color: #FFFFFF;
    padding: 1rem;
}
[data-testid="stSidebarHeader"] {
    color: #FFFFFF;
    font-size: 22px;
    font-weight: bold;
    margin-bottom: 20px;
}
[data-testid="stSidebar"] p {
    color: #FFFFFF;
}
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] input {
    color: #000000;
}
h1, h2, h3 {
    color: #003366;
    font-weight: bold;
}
.stButton > button {
    color: #003366;
    border: 2px solid #003366;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 16px;
}
.stButton > button:hover {
    background-color: #e6f0ff;
    color: #003366;
    border: 2px solid #003366;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

st.markdown(
    """
    <h1 style='text-align: center; width: 100%; font-size: 3em; margin-top: 20px;'>
       E-Commerce Return Analysis Dashboard
    </h1>
    <hr>
    """,
    unsafe_allow_html=True
)

# --- Simulate dataset ---
np.random.seed(42)
n = 2000

categories = ['Women > Ethnic Wear', 'Men > Casual', 'Kids > Wear', 'Electronics', 'Home & Kitchen', 'Sports']
brands = ['Ziyaa', 'Roadster', 'Local Seller', 'H&M', 'Samsung', 'Nike', 'Adidas', 'LG', 'Sony', 'Philips']

prices = np.concatenate([
    np.random.choice(range(100, 500), size=int(n*0.3)),
    np.random.choice(range(500, 1500), size=int(n*0.4)),
    np.random.choice(range(1500, 10000), size=int(n*0.3))
])

product_names_map = {
    'Women > Ethnic Wear': ['Kurti', 'Saree', 'Lehenga'],
    'Men > Casual': ['T-Shirt', 'Jeans', 'Shirt'],
    'Kids > Wear': ['Kids T-Shirt', 'Kids Jeans', 'Kids Jacket'],
    'Electronics': ['Smartphone', 'Laptop', 'Headphones'],
    'Home & Kitchen': ['Mixer', 'Cookware Set', 'Vacuum Cleaner'],
    'Sports': ['Football', 'Tennis Racket', 'Yoga Mat']
}

category_choices = np.random.choice(categories, size=n)
product_names = [np.random.choice(product_names_map[cat]) for cat in category_choices]

sizes = []
for cat in category_choices:
    if 'Wear' in cat or cat == 'Men > Casual' or cat == 'Sports':
        sizes.append(np.random.choice(['S', 'M', 'L', 'XL']))
    else:
        sizes.append(None)

brand_choices = np.random.choice(brands, size=n)

return_status = []
for cat in category_choices:
    if cat == 'Women > Ethnic Wear':
        return_status.append(np.random.choice(['Returned', 'Not Returned'], p=[0.60, 0.40]))
    elif cat == 'Electronics':
        return_status.append(np.random.choice(['Returned', 'Not Returned'], p=[0.50, 0.50]))
    else:
        return_status.append(np.random.choice(['Returned', 'Not Returned'], p=[0.30, 0.70]))

reasons = ['Size issue', 'Quality', 'Not as shown', 'Other']
return_reasons = [np.random.choice(reasons) if status == 'Returned' else 'NA' for status in return_status]

order_dates = pd.to_datetime('2024-01-01') + pd.to_timedelta(np.random.randint(0, 180, size=n), unit='D')

tech_reviews = [
    "Stopped working after one week.",
    "Device heats up too much.",
    "Battery life is terrible.",
    "Not charging properly.",
    "Came with scratches.",
    "Packaging was damaged.",
    "Doesn't connect via Bluetooth.",
    "Poor sound quality.",
    "Not turning on.",
    "Defective screen.",
    "The quality is bad for the price.",
    "Affordable and great!",
    "Excellent product, highly recommend!",
    "Good value for money.",
    "Did not meet expectations."
]
general_reviews = [
    "The quality is bad for the price.",
    "Product was torn.",
    "Fitting is not proper.",
    "Very good, loved it!",
    "Loose fitting but good material.",
    "Too costly for what it offers.",
    "Received wrong size.",
    "Affordable and great!",
    "Poor stitching quality.",
    "Looks different than image.",
    "Excellent product, highly recommend!",
    "Did not meet expectations.",
    "Good value for money.",
    "Color was not as shown."
]

customer_reviews = []
for cat in category_choices:
    if cat == 'Electronics':
        customer_reviews.append(np.random.choice(tech_reviews))
    else:
        customer_reviews.append(np.random.choice(general_reviews))

df = pd.DataFrame({
    'order_id': np.arange(1, n+1),
    'product_name': product_names,
    'category': category_choices,
    'price': prices,
    'size': sizes,
    'brand': brand_choices,
    'return_status': return_status,
    'return_reason': return_reasons,
    'order_date': order_dates,
    'customer_review': customer_reviews
})

# --- Sentiment Analysis ---
# df['sentiment'] = df['customer_review'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
def analyze_sentiment_batch(texts, batch_size=32):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        results.extend(sentiment_analyzer(batch))
    return [r['label'] for r in results]

df['sentiment'] = analyze_sentiment_batch(df['customer_review'].tolist())

# --- Preprocessing ---
df['price_bucket'] = pd.cut(df['price'], bins=[0, 500, 1000, 20000], labels=["<₹500", "₹500–1000", ">₹1000"])
df['order_month'] = df['order_date'].dt.to_period('M').dt.to_timestamp()

# --- Sidebar ---
st.sidebar.header("Filter Options")
selected_category = st.sidebar.multiselect("Select Category", options=df['category'].unique(), default=df['category'].unique())
selected_brand = st.sidebar.multiselect("Select Brand", options=df['brand'].unique(), default=df['brand'].unique())

filtered_df = df[(df['category'].isin(selected_category)) & (df['brand'].isin(selected_brand))]

# --- Return Rate by Category ---
return_by_cat = (
    filtered_df.groupby('category')['return_status']
    .apply(lambda x: (x == 'Returned').mean() * 100)
    .reset_index(name='Return %')
)

# Ensure 'Return %' is float
return_by_cat['Return %'] = return_by_cat['Return %'].astype(float)

# Optional: create formatted text column for labels (fallback)
return_by_cat['Return_text'] = return_by_cat['Return %'].map('{:.2f}%'.format)

st.subheader("📊 Return Rate by Category")
fig_cat = px.bar(
    return_by_cat,
    x='category',
    y='Return %',
    text='Return %',  # or 'Return_text' to bypass texttemplate if needed
    color='Return %',
    color_continuous_scale='Reds',
    range_y=[0, 100]
)

fig_cat.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
st.plotly_chart(fig_cat, use_container_width=True)

# --- Return Rate by Price Bucket ---
price_return_rate = (
    filtered_df.groupby('price_bucket')['return_status']
    .apply(lambda x: (x == 'Returned').mean() * 100)
    .reset_index(name='Return %')
)

price_return_rate['Return %'] = price_return_rate['Return %'].astype(float)

# Optional: create formatted text column for labels
price_return_rate['Return_text'] = price_return_rate['Return %'].map('{:.2f}%'.format)

st.subheader("💰 Return Rate by Price Bucket")
fig_price = px.bar(
    price_return_rate,
    x='price_bucket',
    y='Return %',
    text='Return %',  # or 'Return_text' if you prefer preformatted labels
    color='Return %',
    color_continuous_scale='Blues',
    range_y=[0, 100]
)

fig_price.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
st.plotly_chart(fig_price, use_container_width=True)

# --- Monthly Return Trend ---
monthly_returns = (
    filtered_df.groupby('order_month')['return_status']
    .apply(lambda x: (x == 'Returned').mean() * 100)
    .reset_index(name='Return %')
)

monthly_returns['Return %'] = monthly_returns['Return %'].astype(float)

st.subheader("📅 Monthly Return Rate Trend")
fig_ts = px.line(monthly_returns, x='order_month', y='Return %', markers=True)
fig_ts.update_layout(yaxis_range=[0, 100])
st.plotly_chart(fig_ts, use_container_width=True)

# --- Alerts ---
alert_threshold = 30
high_return_cats = return_by_cat[return_by_cat['Return %'] > alert_threshold]
high_return_prices = price_return_rate[price_return_rate['Return %'] > alert_threshold]

if not high_return_cats.empty or not high_return_prices.empty:
    st.error("⚠️ High Return Rate Alert!")
    if not high_return_cats.empty:
        st.write("Categories with return rate above 30%:")
        st.dataframe(high_return_cats)
    if not high_return_prices.empty:
        st.write("Price buckets with return rate above 30%:")
        st.dataframe(high_return_prices)
else:
    st.success("✅ All return rates are within normal limits.")

# --- Sentiment Insights ---
st.subheader("🧠 Customer Sentiment Insights")
sentiment_counts = df['sentiment'].value_counts()
st.write("### Sentiment Distribution:")
st.write(sentiment_counts)

negative_reviews = df[df['sentiment'] == 'NEGATIVE']
common_issues = negative_reviews['customer_review'].value_counts().head(5)
st.write("### Top Issues from Negative Reviews:")
st.table(common_issues)

# --- Smart Advice Logic ---
def get_sentiment_advice(sentiment_counts, negative_reviews, threshold=0.3):
    negative_ratio = sentiment_counts.get('NEGATIVE', 0) / sentiment_counts.sum()
    if negative_ratio > threshold and not negative_reviews.empty:
        top_issue = negative_reviews['customer_review'].value_counts().idxmax()
        return f"🚨 High negative sentiment detected — Investigate top complaint: **{top_issue}**."
    elif negative_ratio > threshold:
        return "🚨 High negative sentiment detected — consider detailed feedback review."
    else:
        return "🟢 Sentiment is healthy. No immediate action needed."

advice_text = get_sentiment_advice(sentiment_counts, negative_reviews)
st.info(advice_text)

# --- Download Button ---
def convert_df_to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Return Data')
    return output.getvalue()

st.subheader("📥 Download Filtered Data")
excel_data = convert_df_to_excel(filtered_df)
st.download_button(
    label="Download Data as Excel",
    data=excel_data,
    file_name="filtered_return_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
