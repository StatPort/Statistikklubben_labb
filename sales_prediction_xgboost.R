### NOTE. 
# A few things worth noting:
# Label encoding in R won't produce identical integer mappings as Python's LabelEncoder — the ordering depends on factor levels. If you need exact parity with the Python outputs, you'd need to hardcode the same mapping.
# Train/test split uses caret::createDataPartition which stratifies by Y, vs Python's random split. Results may differ slightly — swap for a plain sample() if you want a closer match.
# R² is computed as cor(actual, predicted)^2 since base R doesn't have a built-in equivalent to sklearn's r2_score (they're mathematically identical for regression).


# ============================================================
# Sales Prediction with XGBoost
# Translated from Python/Jupyter notebook
# ============================================================

library(tidyverse)
library(xgboost)
library(caret)

# ============================================================
# LOAD DATA
# ============================================================
sales_data <- read_csv("data.csv")
head(sales_data)
dim(sales_data)
str(sales_data)
summary(sales_data)

# ============================================================
# MISSING VALUES
# ============================================================
colSums(is.na(sales_data))

# Fill Item_Weight with mean
sales_data <- sales_data %>%
  mutate(Item_Weight = ifelse(is.na(Item_Weight), mean(Item_Weight, na.rm = TRUE), Item_Weight))

# Fill Outlet_Size with mode per Outlet_Type
mode_val <- function(x) {
  ux <- na.omit(x)
  ux[which.max(tabulate(match(ux, unique(ux))))]
}

outlet_size_mode <- sales_data %>%
  group_by(Outlet_Type) %>%
  summarise(mode_size = mode_val(Outlet_Size))

sales_data <- sales_data %>%
  left_join(outlet_size_mode, by = "Outlet_Type") %>%
  mutate(Outlet_Size = ifelse(is.na(Outlet_Size), mode_size, Outlet_Size)) %>%
  select(-mode_size)

colSums(is.na(sales_data))

# ============================================================
# DATA VISUALIZATION
# ============================================================

# Numerical features
ggplot(sales_data, aes(x = Item_Weight)) +
  geom_histogram(fill = "purple", bins = 30) +
  labs(title = "Item Weight Distribution")

ggplot(sales_data, aes(x = Item_Visibility)) +
  geom_histogram(fill = "purple", bins = 30) +
  labs(title = "Item Visibility Distribution")

ggplot(sales_data, aes(x = Item_MRP)) +
  geom_histogram(fill = "purple", bins = 30) +
  labs(title = "Item MRP Distribution")

ggplot(sales_data, aes(x = Item_Outlet_Sales)) +
  geom_histogram(fill = "purple", bins = 30) +
  labs(title = "Item Outlet Sales Distribution")

# Categorical features
ggplot(sales_data, aes(x = factor(Outlet_Establishment_Year))) +
  geom_bar(fill = "purple") +
  labs(title = "Outlet Establishment Year", x = "Year")

ggplot(sales_data, aes(x = Item_Fat_Content)) +
  geom_bar(fill = "purple") +
  labs(title = "Item Fat Content")

ggplot(sales_data, aes(x = Item_Type)) +
  geom_bar(fill = "purple") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Item Type")

ggplot(sales_data, aes(x = Outlet_Size)) +
  geom_bar(fill = "purple") +
  labs(title = "Outlet Size")

# ============================================================
# PREPROCESSING
# ============================================================

# Fix inconsistent labels in Item_Fat_Content
sales_data <- sales_data %>%
  mutate(Item_Fat_Content = recode(Item_Fat_Content,
    "low fat" = "Low Fat",
    "LF"      = "Low Fat",
    "reg"     = "Regular"
  ))

# Label encoding (factor -> integer)
encode <- function(x) as.integer(factor(x)) - 1L

sales_data <- sales_data %>%
  mutate(
    Item_Identifier      = encode(Item_Identifier),
    Item_Fat_Content     = encode(Item_Fat_Content),
    Item_Type            = encode(Item_Type),
    Outlet_Identifier    = encode(Outlet_Identifier),
    Outlet_Size          = encode(Outlet_Size),
    Outlet_Location_Type = encode(Outlet_Location_Type),
    Outlet_Type          = encode(Outlet_Type)
  )

# ============================================================
# FEATURES & TARGET
# ============================================================
X <- sales_data %>% select(-Item_Outlet_Sales, -Item_Identifier)
Y <- sales_data$Item_Outlet_Sales

# ============================================================
# TRAIN / TEST SPLIT (80/20)
# ============================================================
set.seed(2)
train_idx <- createDataPartition(Y, p = 0.8, list = FALSE)

X_train <- X[train_idx, ]
X_test  <- X[-train_idx, ]
Y_train <- Y[train_idx]
Y_test  <- Y[-train_idx]

cat("Full:", nrow(X), "| Train:", nrow(X_train), "| Test:", nrow(X_test), "\n")

# ============================================================
# XGBOOST MODEL
# ============================================================
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = Y_train)
dtest  <- xgb.DMatrix(data = as.matrix(X_test),  label = Y_test)

params <- list(
  objective   = "reg:squarederror",
  max_depth   = 6,
  eta         = 0.05,   # learning_rate
  seed        = 42
)

regressor <- xgb.train(
  params  = params,
  data    = dtrain,
  nrounds = 500
)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
importance <- xgb.importance(model = regressor, feature_names = colnames(X_train))
xgb.plot.importance(importance, main = "Feature Importance")

# ============================================================
# EVALUATION
# ============================================================
# Train R²
train_pred <- predict(regressor, dtrain)
r2_train <- cor(Y_train, train_pred)^2
cat("Train R² =", round(r2_train, 4), "\n")

# Test R²
test_pred <- predict(regressor, dtest)
r2_test <- cor(Y_test, test_pred)^2
cat("Test R²  =", round(r2_test, 4), "\n")

# ============================================================
# PREDICTIVE SYSTEM (first product example)
# ============================================================
input_data <- matrix(
  c(9.300, 0, 0.016047, 4, 249.8092, 9, 1999, 1, 0, 1),
  nrow = 1
)
colnames(input_data) <- colnames(X_train)

prediction <- predict(regressor, xgb.DMatrix(input_data))
cat("Predicted sales:", prediction, "\n")
