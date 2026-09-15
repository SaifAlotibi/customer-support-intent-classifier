# Customer Support Intent Classifier

An end-to-end NLP machine learning project that classifies customer support messages into 27 different customer intents using TF-IDF and Linear Support Vector Machine (SVM).

The project includes data preprocessing, text feature extraction, model comparison, error analysis, and deployment through FastAPI and Streamlit.

## Project Overview

Customer support systems receive large numbers of messages that need to be categorized before they can be routed or handled automatically.

This project builds a multiclass text classification system that takes a customer message as input and predicts its intent.

**Example:**

```text
Input:
"I forgot my password and cannot access my account"

Prediction:
recover_password
```

## Dataset

The project uses the **Bitext Customer Support LLM Chatbot Training Dataset**.

The original dataset contains:

* 26,872 customer support messages
* 27 intent classes
* Customer instructions and corresponding intents
* Example responses and metadata

For this project, only the `instruction` column was used as the input feature and `intent` as the target.

The `category`, `response`, and `flags` columns were not used as model features.

### Data Cleaning

Exact duplicate customer messages were removed before splitting the dataset.

```text
Original samples: 26,872
After removing duplicates: 24,635
Duplicates removed: 2,237
```

The data was then divided using a stratified train/test split:

```text
Training: 19,708
Testing:   4,927
```

## Machine Learning Approach

### 1. TF-IDF

Customer messages were converted from text into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

Unigrams and bigrams were used:

```python
TfidfVectorizer(
    ngram_range=(1, 2)
)
```

This produced:

```text
11,298 TF-IDF features
```

### 2. Model Comparison

Two supervised classification models were evaluated:

* Logistic Regression
* Linear Support Vector Machine (SVM)

### Results

| Model               |   Accuracy |   Macro F1 |
| ------------------- | ---------: | ---------: |
| Logistic Regression |     98.96% |     98.91% |
| Linear SVM          | **99.39%** | **99.37%** |

Linear SVM was selected as the final model because it achieved the best performance.

## Error Analysis

The final model made only **30 errors out of 4,927 test messages**.

The most common confusions occurred between semantically similar intents:

* `create_account` → `delete_account`
* `get_invoice` → `check_invoice`
* `contact_customer_service` → `contact_human_agent`
* `cancel_order` → `change_order`
* `change_order` → `place_order`

Many
