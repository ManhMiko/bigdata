import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd
import numpy as np
from IPython.display import HTML, display
from wordcloud import WordCloud
import sys
import time
import webbrowser

# Liệt kê file trong thư mục
for dirname, _, filenames in os.walk(r"D:\big data\kaggle"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Đọc file CSV một lần
FILE_PATH = r"D:\big data\kaggle\All Electronics.csv"
df = pd.read_csv(FILE_PATH)
pd.set_option('max_colwidth', 400)

# Hiển thị thông tin cơ bản
print("Initial dataset shape:", df.shape)
print(df.head(3).to_string())
print("Unique main_categories:", df['main_category'].unique())
print("Unique sub_categories:", df['sub_category'].unique())
df.info()

# Xuất HTML cho 10 dòng đầu
html_data = df.head(10).to_html()
with open("output.html", "w", encoding="utf-8") as f:
    f.write(html_data)
webbrowser.open("output.html")

# Hàm làm sạch dữ liệu số
def clean_numeric_column(series, allow_decimal=False):
    pattern = r'[^\d.]' if allow_decimal else r'[^\d]'
    return (
        series.astype(str)
        .str.replace(pattern, '', regex=True)
        .replace('', np.nan)
        .astype(float)
    )

columns_to_clean = {
    'ratings': True,
    'no_of_ratings': False,
    'discount_price': False,
    'actual_price': False
}

for col, allow_decimal in columns_to_clean.items():
    df[f'{col}_cleaned'] = clean_numeric_column(df[col], allow_decimal)

# Thống kê dữ liệu đã làm sạch
print(df[[f'{col}_cleaned' for col in columns_to_clean]].describe())
top_10_rated = df.sort_values(by='no_of_ratings_cleaned', ascending=False).head(10)
display(top_10_rated[['name', 'no_of_ratings', 'ratings', 'discount_price', 'actual_price']])

# Trực quan hóa
numeric_cols = ['ratings_cleaned', 'no_of_ratings_cleaned', 'discount_price_cleaned', 'actual_price_cleaned']
df[numeric_cols].hist(bins=30, figsize=(12, 8), color='teal', edgecolor='black')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['discount_price_cleaned'], y=df['ratings_cleaned'], alpha=0.7, color='purple')
plt.xlabel("Discount Price (Cleaned)")
plt.ylabel("Ratings (Cleaned)")
plt.title("Price vs. Ratings")
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['no_of_ratings_cleaned'], y=df['ratings_cleaned'], alpha=0.6, color='green')
plt.xscale('log')
plt.xlabel("Number of Ratings (log scale)")
plt.ylabel("Star Rating")
plt.title("Star Rating vs. Number of Ratings")
plt.show()

# Tính phần trăm giảm giá
df['discount_diff'] = df['actual_price_cleaned'] - df['discount_price_cleaned']
df['discount_pct'] = (df['discount_diff'] / df['actual_price_cleaned']) * 100
df['discount_pct'] = df['discount_pct'].fillna(0).clip(lower=0, upper=100)

plt.figure(figsize=(7, 5))
sns.histplot(df['discount_pct'], bins=20, color='gold', kde=True)
plt.title("Histogram of Discount Percentage")
plt.xlabel("Discount %")
plt.show()

# Ma trận tương quan
num_df = df[['ratings_cleaned', 'no_of_ratings_cleaned', 'discount_price_cleaned', 'actual_price_cleaned', 'discount_pct']]
corr_matrix = num_df.corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# WordCloud
all_names = " ".join(df['name'].astype(str).fillna(""))
stop_words = set(stopwords.words('english')).union({'with', 'for', 'and', 'in', 'to', 'of', 'the', 'by'})
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stop_words).generate(all_names)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud of Product Names")
plt.show()

# Làm sạch dữ liệu cho hệ thống gợi ý
drop_cols = ['main_category', 'sub_category']
df.drop(drop_cols, axis=1, inplace=True)

duplicates_count = df.duplicated().sum()
print("Number of duplicated rows:", duplicates_count)
if duplicates_count > 0:
    df.drop_duplicates(inplace=True)

df.dropna(subset=['name'], inplace=True)
print("Shape after cleaning:", df.shape)

df['search_terms'] = df['name'].str.lower().str.replace(r'[^\w\d\s]+', ' ', regex=True)
stemmer = PorterStemmer()
df['search_terms'] = df['search_terms'].apply(lambda x: " ".join(stemmer.stem(word) for word in x.split()))

# Tạo ma trận đặc trưng
vectorizer = CountVectorizer(max_features=5000, stop_words='english', dtype=np.int8)
feature_matrix = vectorizer.fit_transform(df['search_terms'])
print("Feature matrix shape:", feature_matrix.shape)

similarities = cosine_similarity(feature_matrix, dense_output=False)
print("Similarity matrix shape:", similarities.shape)
print(f"Size of the similarity matrix: {sys.getsizeof(similarities) / 1024 / 1024:.2f} MB")

# Tìm top K láng giềng
def build_top_neighbors_matrix(sim_matrix, k=10):
    top_neighbors_list = []
    for row_idx in range(sim_matrix.shape[0]):
        sim_scores = list(enumerate(sim_matrix[row_idx].toarray()[0]))
        sim_scores_sorted = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_neighbors = sim_scores_sorted[1:k + 1]
        top_neighbors_indices = [item[0] for item in top_neighbors]
        top_neighbors_list.append(top_neighbors_indices)
    return np.array(top_neighbors_list, dtype=np.int32)

start_time = time.time()
top_k_neighbors = build_top_neighbors_matrix(similarities, k=10)
end_time = time.time()
print(f"Time to build top-10 neighbors structure: {end_time - start_time:.3f} sec")
print("Shape of top_k_neighbors:", top_k_neighbors.shape)
print(f"Memory usage of top_k_neighbors: {sys.getsizeof(top_k_neighbors) / 1024 / 1024:.2f} MB")
del similarities

# Xử lý URL
def shorten_image_url(url_str):
    url_str = str(url_str)
    part_after = url_str.split('images/')[-1].split('._AC_UL320_.jpg')[0]
    return part_after

df['image_id'] = df['image'].apply(shorten_image_url)

def shorten_amazon_link(link_str):
    link_str = str(link_str)
    return link_str.replace('https://www.amazon.in/', '')

df['link_id'] = df['link'].apply(shorten_amazon_link)

# Hệ thống gợi ý
name_to_idx_map = {name: i for i, name in enumerate(df['name'])}

def find_index_by_name(product_name):
    return name_to_idx_map.get(product_name, -1)

def find_index_by_partial_link(partial_link):
    mask = df['link_id'].str.contains(partial_link, na=False)
    matches = df[mask]
    return matches.index[0] if not matches.empty else -1

def get_similar_products(product_query, k=5):
    idx = find_index_by_name(product_query)
    if idx == -1:
        idx = find_index_by_partial_link(product_query)
    if idx == -1:
        print(f"Product '{product_query}' not found in the dataset.")
        return pd.DataFrame()
    neighbors_indices = top_k_neighbors[idx][:k]
    rec_df = df.iloc[neighbors_indices][['name', 'ratings', 'no_of_ratings', 'discount_price', 'actual_price', 'link', 'image']]
    rec_df.reset_index(drop=True, inplace=True)
    return rec_df

# Ví dụ 1: Dell Mouse
some_product_1 = "Dell MS116 1000Dpi USB Wired Optical Mouse, Led Tracking, Scrolling Wheel, Plug and Play"
print("::---SELECTED PRODUCT 1---::")
print(some_product_1)

print("\n::---TOP 5 RECOMMENDATIONS FOR PRODUCT 1---::")
recommendations_df_1 = get_similar_products(some_product_1, k=5)
selected_idx_1 = find_index_by_name(some_product_1)
selected_row_1 = df.iloc[selected_idx_1]
selected_name_1 = selected_row_1['name']
selected_price_1 = selected_row_1['discount_price']
selected_image_url_1 = selected_row_1['image']

# Tạo chuỗi HTML cho ví dụ 1
html_content_1 = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Product Recommendations - Dell Mouse</title>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .container {{ max-width: 800px; margin: 20px auto; }}
        .product {{ display: flex; align-items: center; margin-bottom: 20px; }}
        .recommendation {{ border: 1px solid #ddd; margin-bottom: 15px; padding: 10px; display: flex; align-items: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Selected Product</h2>
        <div class="product">
            <img src="{selected_image_url_1}" alt="Product Image" style="width: 120px; margin-right: 20px; border: 1px solid #ccc;"/>
            <div>
                <h4 style="margin: 0;">{selected_name_1}</h4>
                <p style="margin: 5px 0;">Price: {selected_price_1}</p>
            </div>
        </div>
        <h3>Top 5 Recommendations</h3>
"""

for _, row in recommendations_df_1.iterrows():
    rec_name = row['name']
    rec_price = row['discount_price']
    rec_image = row['image']
    html_content_1 += f"""
        <div class="recommendation">
            <img src="{rec_image}" alt="Recommended Product" style="width: 80px; margin-right: 20px; border: 1px solid #ccc;"/>
            <div>
                <p style="margin: 0;"><strong>{rec_name}</strong></p>
                <p style="margin: 5px 0;">Price: {rec_price}</p>
            </div>
        </div>
    """

html_content_1 += """
    </div>
</body>
</html>
"""

# Ghi và mở file HTML cho ví dụ 1
with open("recommendations_dell.html", "w", encoding="utf-8") as f:
    f.write(html_content_1)
webbrowser.open("recommendations_dell.html")

# Ví dụ 2: OnePlus Nord
some_product_2 = "OnePlus Nord CE 2 Lite 5G (Black Dusk, 6GB RAM, 128GB Storage)"
print("::---SELECTED PRODUCT 2---::")
print(some_product_2)

print("\n::---TOP 5 RECOMMENDATIONS FOR PRODUCT 2---::")
recommendations_df_2 = get_similar_products(some_product_2, k=5)
selected_idx_2 = find_index_by_name(some_product_2)
selected_row_2 = df.iloc[selected_idx_2]
selected_name_2 = selected_row_2['name']
selected_price_2 = selected_row_2['discount_price']
selected_image_url_2 = selected_row_2['image']

# Tạo chuỗi HTML cho ví dụ 2
html_content_2 = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Product Recommendations - OnePlus Nord</title>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .container {{ max-width: 800px; margin: 20px auto; }}
        .product {{ display: flex; align-items: center; margin-bottom: 20px; }}
        .recommendation {{ border: 1px solid #ddd; margin-bottom: 15px; padding: 10px; display: flex; align-items: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h2>Selected Product</h2>
        <div class="product">
            <img src="{selected_image_url_2}" alt="Product Image" style="width: 120px; margin-right: 20px; border: 1px solid #ccc;"/>
            <div>
                <h4 style="margin: 0;">{selected_name_2}</h4>
                <p style="margin: 5px 0;">Price: {selected_price_2}</p>
            </div>
        </div>
        <h3>Top 5 Recommendations</h3>
"""

for _, row in recommendations_df_2.iterrows():
    rec_name = row['name']
    rec_price = row['discount_price']
    rec_image = row['image']
    html_content_2 += f"""
        <div class="recommendation">
            <img src="{rec_image}" alt="Recommended Product" style="width: 80px; margin-right: 20px; border: 1px solid #ccc;"/>
            <div>
                <p style="margin: 0;"><strong>{rec_name}</strong></p>
                <p style="margin: 5px 0;">Price: {rec_price}</p>
            </div>
        </div>
    """

html_content_2 += """
    </div>
</body>
</html>
"""

# Ghi và mở file HTML cho ví dụ 2
with open("recommendations_oneplus.html", "w", encoding="utf-8") as f:
    f.write(html_content_2)
webbrowser.open("recommendations_oneplus.html")

input("Press Enter to exit...")