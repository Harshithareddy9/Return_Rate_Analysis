import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from transformers import pipeline

# --- Page config ---
st.set_page_config(page_title="E-Commerce Returns", layout='wide')

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_analyzer = load_sentiment_model()

# --- Full Custom CSS ---
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
n = 5000  # Reduced dataset size for faster processing

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

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")
selected_category = st.sidebar.multiselect("Select Category", options=df['category'].unique(), default=df['category'].unique())
selected_brand = st.sidebar.multiselect("Select Brand", options=df['brand'].unique(), default=df['brand'].unique())

run_sentiment = st.sidebar.checkbox("🔍 Run Sentiment Analysis on Reviews")

filtered_df = df[(df['category'].isin(selected_category)) & (df['brand'].isin(selected_brand))]

# --- Sentiment Analysis (only run if user checks box) ---
if run_sentiment:
    with st.spinner("Analyzing sentiment..."):
        filtered_df['sentiment'] = filtered_df['customer_review'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
else:
    filtered_df['sentiment'] = 'Not analyzed'

# --- Preprocessing ---
filtered_df['price_bucket'] = pd.cut(filtered_df['price'], bins=[0, 500, 1000, 20000], labels=["<₹500", "₹500–1000", ">₹1000"])
filtered_df['order_month'] = filtered_df['order_date'].dt.to_period('M').dt.to_timestamp()

# --- Return Rate by Category ---
return_by_cat = filtered_df.groupby('category')['return_status'].apply(lambda x: (x == 'Returned').mean() * 100).reset_index(name='Return %')
st.subheader("📊 Return Rate by Category")
fig_cat = px.bar(return_by_cat, x='category', y='Return %', text='Return %',
                 labels={'Return %': 'Return Rate (%)', 'category': 'Category'}, color='Return %',
                 color_continuous_scale='Reds', range_y=[0, 100])
fig_cat.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
st.plotly_chart(fig_cat, use_container_width=True)

# --- Return Rate by Price Bucket ---
price_return_rate = filtered_df.groupby('price_bucket')['return_status'].apply(lambda x: (x == 'Returned').mean() * 100).reset_index(name='Return %')
st.subheader("💰 Return Rate by Price Bucket")
fig_price = px.bar(price_return_rate, x='price_bucket', y='Return %', text='Return %',
                   labels={'Return %': 'Return Rate (%)', 'price_bucket': 'Price Bucket'}, color='Return %',
                   color_continuous_scale='Blues', range_y=[0, 100])
fig_price.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
st.plotly_chart(fig_price, use_container_width=True)

# --- Monthly Return Trend ---
monthly_returns = filtered_df.groupby('order_month')['return_status'].apply(lambda x: (x == 'Returned').mean() * 100).reset_index(name='Return %')
st.subheader("📅 Monthly Return Rate Trend")
fig_ts = px.line(monthly_returns, x='order_month', y='Return %',
                 labels={'order_month': 'Month', 'Return %': 'Return Rate (%)'},
                 markers=True)
fig_ts.update_layout(yaxis_range=[0, 100])
st.plotly_chart(fig_ts, use_container_width=True)

# --- Warning Section ---
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
