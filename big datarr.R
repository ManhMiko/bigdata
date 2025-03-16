# Cài đặt các gói cần thiết (chỉ chạy lần đầu nếu chưa cài)
# install.packages(c("tidyverse", "text2vec", "htmltools", "wordcloud", "tm", "SnowballC"))

# Load các thư viện
library(tidyverse)      # Thay thế pandas, seaborn
library(text2vec)       # Thay thế sklearn.feature_extraction.text cho cosine similarity
library(htmltools)      # Thay thế IPython.display.HTML để tạo HTML
library(wordcloud)      # Tạo word cloud
library(tm)             # Xử lý văn bản (stopwords)
library(SnowballC)      # Stemming (thay thế PorterStemmer)

# Liệt kê file trong thư mục
dir_path <- "D:/big data/kaggle"
files <- list.files(dir_path, recursive = TRUE, full.names = TRUE)
for (file in files) {
  cat(file, "\n")
}

# Đọc file CSV
file_path <- "D:/big data/kaggle/All Electronics.csv"
df <- read_csv(file_path)

# Thiết lập tùy chọn hiển thị
options(max.print = 400)  # Tương tự pd.set_option('max_colwidth', 400)

# Hiển thị thông tin cơ bản
cat("Initial dataset shape:", dim(df), "\n")
print(head(df, 3))
cat("Unique main_categories:", unique(df$main_category), "\n")
cat("Unique sub_categories:", unique(df$sub_category), "\n")
str(df)

# Xuất HTML cho 10 dòng đầu
html_data <- knitr::kable(head(df, 10), format = "html")
write_file(html_data, "output.html")
browseURL("output.html")

# Hàm làm sạch dữ liệu số
clean_numeric_column <- function(series, allow_decimal = FALSE) {
  pattern <- if (allow_decimal) "[^0-9.]" else "[^0-9]"
  series <- as.character(series)
  series <- gsub(pattern, "", series)
  series[series == ""] <- NA
  return(as.numeric(series))
}

# Áp dụng làm sạch cho các cột số
columns_to_clean <- list(
  ratings = TRUE,
  no_of_ratings = FALSE,
  discount_price = FALSE,
  actual_price = FALSE
)

for (col in names(columns_to_clean)) {
  df[[paste0(col, "_cleaned")]] <- clean_numeric_column(df[[col]], columns_to_clean[[col]])
}

# Thống kê dữ liệu đã làm sạch
cleaned_cols <- paste0(names(columns_to_clean), "_cleaned")
print(summary(df[cleaned_cols]))
top_10_rated <- df %>% 
  arrange(desc(no_of_ratings_cleaned)) %>% 
  head(10) %>% 
  select(name, no_of_ratings, ratings, discount_price, actual_price)
print(top_10_rated)

# Trực quan hóa
numeric_cols <- c("ratings_cleaned", "no_of_ratings_cleaned", "discount_price_cleaned", "actual_price_cleaned")

# Histogram
par(mfrow = c(2, 2))  # Sắp xếp 4 biểu đồ thành lưới 2x2
for (col in numeric_cols) {
  hist(df[[col]], breaks = 30, col = "teal", border = "black", main = col, xlab = col)
}
dev.off()

# Scatter Plot: Giá giảm vs Điểm đánh giá
ggplot(df, aes(x = discount_price_cleaned, y = ratings_cleaned)) +
  geom_point(alpha = 0.7, color = "purple") +
  labs(x = "Discount Price (Cleaned)", y = "Ratings (Cleaned)", title = "Price vs. Ratings") +
  theme_minimal()
ggsave("price_vs_ratings.png")

# Scatter Plot: Số lượng đánh giá vs Điểm đánh giá
ggplot(df, aes(x = no_of_ratings_cleaned, y = ratings_cleaned)) +
  geom_point(alpha = 0.6, color = "green") +
  scale_x_log10() +
  labs(x = "Number of Ratings (log scale)", y = "Star Rating", title = "Star Rating vs. Number of Ratings") +
  theme_minimal()
ggsave("ratings_vs_no_of_ratings.png")

# Tính phần trăm giảm giá
df <- df %>%
  mutate(discount_diff = actual_price_cleaned - discount_price_cleaned,
         discount_pct = (discount_diff / actual_price_cleaned) * 100) %>%
  mutate(discount_pct = replace(discount_pct, is.na(discount_pct), 0)) %>%
  mutate(discount_pct = pmax(pmin(discount_pct, 100), 0))

# Histogram của phần trăm giảm giá
ggplot(df, aes(x = discount_pct)) +
  geom_histogram(bins = 20, fill = "gold", color = "black") +
  geom_density(alpha = 0.2) +
  labs(title = "Histogram of Discount Percentage", x = "Discount %") +
  theme_minimal()
ggsave("discount_pct_histogram.png")

# Ma trận tương quan
num_df <- df %>% select(ratings_cleaned, no_of_ratings_cleaned, discount_price_cleaned, actual_price_cleaned, discount_pct)
corr_matrix <- cor(num_df, use = "complete.obs")
corrplot::corrplot(corr_matrix, method = "color", type = "upper", addCoef.col = "black", tl.col = "black", 
                   col = viridis::viridis(100), number.digits = 2, title = "Correlation Heatmap")

# WordCloud
all_names <- paste(df$name, collapse = " ")
stop_words <- c(stopwords("en"), "with", "for", "and", "in", "to", "of", "the", "by")
wordcloud(words = all_names, min.freq = 1, scale = c(3, 0.5), colors = brewer.pal(8, "Dark2"), 
          random.order = FALSE, stopwords = stop_words, width = 800, height = 400)

# --- PHẦN HỆ THỐNG GỢI Ý SẢN PHẨM ---

# Bước 1: Làm sạch dữ liệu cho hệ thống gợi ý
df <- df %>% select(-main_category, -sub_category)  # Loại bỏ cột không cần thiết
duplicates_count <- sum(duplicated(df))
cat("Number of duplicated rows:", duplicates_count, "\n")
if (duplicates_count > 0) {
  df <- distinct(df)
}
df <- df %>% filter(!is.na(name))
cat("Shape after cleaning:", dim(df), "\n")

# Bước 2: Chuẩn hóa văn bản
df$search_terms <- tolower(df$name)
df$search_terms <- gsub("[^[:alnum:][:space:]]", " ", df$search_terms)
df$search_terms <- sapply(strsplit(df$search_terms, " "), function(words) {
  paste(wordStem(words, language = "english"), collapse = " ")
})

# Bước 3: Tạo ma trận đặc trưng
it <- itoken(df$search_terms, tokenizer = word_tokenizer)
vocab <- create_vocabulary(it, stopwords = stopwords("en"))
vectorizer <- vocab_vectorizer(vocab, grow_dtm = FALSE, skip_grams_window = 0L)
dtm <- create_dtm(it, vectorizer)
cat("Feature matrix shape:", dim(dtm), "\n")

# Bước 4: Tính ma trận độ tương đồng cosin
similarities <- sim2(dtm, method = "cosine", norm = "l2")
cat("Similarity matrix shape:", dim(similarities), "\n")
cat("Size of the similarity matrix:", object.size(similarities) / 1024 / 1024, "MB\n")

# Bước 5: Xây dựng ma trận top K sản phẩm tương tự
build_top_neighbors_matrix <- function(sim_matrix, k = 10) {
  top_neighbors_list <- apply(sim_matrix, 1, function(row) {
    sorted_indices <- order(row, decreasing = TRUE)[2:(k + 1)]  # Bỏ sản phẩm chính nó
    return(sorted_indices - 1)  # Điều chỉnh về index 0-based
  })
  return(t(top_neighbors_list))
}

start_time <- Sys.time()
top_k_neighbors <- build_top_neighbors_matrix(similarities, k = 10)
end_time <- Sys.time()
cat("Time to build top-10 neighbors structure:", difftime(end_time, start_time, units = "secs"), "sec\n")
cat("Shape of top_k_neighbors:", dim(top_k_neighbors), "\n")
cat("Memory usage of top_k_neighbors:", object.size(top_k_neighbors) / 1024 / 1024, "MB\n")

# Bước 6: Xử lý URL
shorten_image_url <- function(url_str) {
  url_str <- as.character(url_str)
  part_after <- sub(".*images/", "", url_str)
  part_after <- sub("\\._AC_UL320_\\.jpg", "", part_after)
  return(part_after)
}

df$image_id <- sapply(df$image, shorten_image_url)

shorten_amazon_link <- function(link_str) {
  link_str <- as.character(link_str)
  return(gsub("https://www.amazon.in/", "", link_str))
}

df$link_id <- sapply(df$link, shorten_amazon_link)

# Bước 7: Xây dựng hệ thống gợi ý
name_to_idx_map <- setNames(0:(nrow(df) - 1), df$name)

find_index_by_name <- function(product_name) {
  idx <- name_to_idx_map[product_name]
  if (is.na(idx)) return(-1)
  return(idx)
}

find_index_by_partial_link <- function(partial_link) {
  mask <- grepl(partial_link, df$link_id, ignore.case = TRUE)
  matches <- which(mask)
  if (length(matches) > 0) return(matches[1] - 1)  # Điều chỉnh về 0-based
  return(-1)
}

get_similar_products <- function(product_query, k = 5) {
  idx <- find_index_by_name(product_query)
  if (idx == -1) {
    idx <- find_index_by_partial_link(product_query)
  }
  if (idx == -1) {
    cat(sprintf("Product '%s' not found in the dataset.\n", product_query))
    return(data.frame())
  }
  neighbors_indices <- top_k_neighbors[idx + 1, 1:k]  # Điều chỉnh về 1-based
  rec_df <- df[neighbors_indices + 1, ] %>% 
    select(name, ratings, no_of_ratings, discount_price, actual_price, link, image)
  row.names(rec_df) <- NULL
  return(rec_df)
}

# Ví dụ 1: Dell Mouse
some_product_1 <- "Dell MS116 1000Dpi USB Wired Optical Mouse, Led Tracking, Scrolling Wheel, Plug and Play"
cat("::---SELECTED PRODUCT 1---::\n", some_product_1, "\n")
cat("\n::---TOP 5 RECOMMENDATIONS FOR PRODUCT 1---::\n")
recommendations_df_1 <- get_similar_products(some_product_1, k = 5)
selected_idx_1 <- find_index_by_name(some_product_1)
selected_row_1 <- df[selected_idx_1 + 1, ]
selected_name_1 <- selected_row_1$name
selected_price_1 <- selected_row_1$discount_price
selected_image_url_1 <- selected_row_1$image

html_content_1 <- tags$html(
  tags$head(
    tags$title("Product Recommendations - Dell Mouse"),
    tags$style(HTML("
      body { font-family: Arial, sans-serif; }
      .container { max-width: 800px; margin: 20px auto; }
      .product { display: flex; align-items: center; margin-bottom: 20px; }
      .recommendation { border: 1px solid #ddd; margin-bottom: 15px; padding: 10px; display: flex; align-items: center; }
    "))
  ),
  tags$body(
    tags$div(class = "container",
             tags$h2("Selected Product"),
             tags$div(class = "product",
                      tags$img(src = selected_image_url_1, alt = "Product Image", style = "width: 120px; margin-right: 20px; border: 1px solid #ccc;"),
                      tags$div(
                        tags$h4(style = "margin: 0;", selected_name_1),
                        tags$p(style = "margin: 5px 0;", paste("Price:", selected_price_1))
                      )
             ),
             tags$h3("Top 5 Recommendations"),
             lapply(1:nrow(recommendations_df_1), function(i) {
               rec_name <- recommendations_df_1$name[i]
               rec_price <- recommendations_df_1$discount_price[i]
               rec_image <- recommendations_df_1$image[i]
               tags$div(class = "recommendation",
                        tags$img(src = rec_image, alt = "Recommended Product", style = "width: 80px; margin-right: 20px; border: 1px solid #ccc;"),
                        tags$div(
                          tags$p(style = "margin: 0;", tags$strong(rec_name)),
                          tags$p(style = "margin: 5px 0;", paste("Price:", rec_price))
                        )
               )
             })
    )
  )
)

save_html(html_content_1, "recommendations_dell.html")
browseURL("recommendations_dell.html")

# Ví dụ 2: OnePlus Nord
some_product_2 <- "OnePlus Nord CE 2 Lite 5G (Black Dusk, 6GB RAM, 128GB Storage)"
cat("::---SELECTED PRODUCT 2---::\n", some_product_2, "\n")
cat("\n::---TOP 5 RECOMMENDATIONS FOR PRODUCT 2---::\n")
recommendations_df_2 <- get_similar_products(some_product_2, k = 5)
selected_idx_2 <- find_index_by_name(some_product_2)
selected_row_2 <- df[selected_idx_2 + 1, ]
selected_name_2 <- selected_row_2$name
selected_price_2 <- selected_row_2$discount_price
selected_image_url_2 <- selected_row_2$image

html_content_2 <- tags$html(
  tags$head(
    tags$title("Product Recommendations - OnePlus Nord"),
    tags$style(HTML("
      body { font-family: Arial, sans-serif; }
      .container { max-width: 800px; margin: 20px auto; }
      .product { display: flex; align-items: center; margin-bottom: 20px; }
      .recommendation { border: 1px solid #ddd; margin-bottom: 15px; padding: 10px; display: flex; align-items: center; }
    "))
  ),
  tags$body(
    tags$div(class = "container",
             tags$h2("Selected Product"),
             tags$div(class = "product",
                      tags$img(src = selected_image_url_2, alt = "Product Image", style = "width: 120px; margin-right: 20px; border: 1px solid #ccc;"),
                      tags$div(
                        tags$h4(style = "margin: 0;", selected_name_2),
                        tags$p(style = "margin: 5px 0;", paste("Price:", selected_price_2))
                      )
             ),
             tags$h3("Top 5 Recommendations"),
             lapply(1:nrow(recommendations_df_2), function(i) {
               rec_name <- recommendations_df_2$name[i]
               rec_price <- recommendations_df_2$discount_price[i]
               rec_image <- recommendations_df_2$image[i]
               tags$div(class = "recommendation",
                        tags$img(src = rec_image, alt = "Recommended Product", style = "width: 80px; margin-right: 20px; border: 1px solid #ccc;"),
                        tags$div(
                          tags$p(style = "margin: 0;", tags$strong(rec_name)),
                          tags$p(style = "margin: 5px 0;", paste("Price:", rec_price))
                        )
               )
             })
    )
  )
)

save_html(html_content_2, "recommendations_oneplus.html")
browseURL("recommendations_oneplus.html")

# Đợi người dùng nhấn Enter để thoát
cat("Press Enter to exit...\n")
invisible(readline())

