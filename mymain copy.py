import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML, display
import random
import re



def logistic_regression_no_penalty(X_train, y_train, X_val, y_val):
    """
    Logistic Regression without penalty for baseline evaluation.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_val: Validation features.
    :param y_val: Validation labels.
    :return: AUC score.
    """
    model = LogisticRegression(penalty='none', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    val_prob = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, val_prob)
    return model, auc_score
def elastic_net_logistic_regression(X_train, y_train, X_val, y_val, alphas=np.linspace(0, 1, 11)):
    """
    Logistic Regression with Elastic Net regularization.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_val: Validation features.
    :param y_val: Validation labels.
    :param alphas: List of alpha values for elastic net (mixing ratio).
    :return: Best model, best alpha, AUC score.
    """
    best_auc = 0
    best_model = None
    best_alpha = None
    
    for alpha in alphas:
        model = LogisticRegressionCV(
            Cs=10,  # Number of C (inverse lambda) values to test
            cv=5,  # 5-fold cross-validation
            penalty='elasticnet',
            solver='saga',
            l1_ratios=[alpha],  # Corresponds to `alpha` in glmnet
            max_iter=1000,
            scoring='roc_auc',
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Evaluate on validation data
        val_prob = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, val_prob)
        
        print(f"Elastic Net Logistic Regression (alpha={alpha:.2f}) AUC: {auc_score:.3f}")
        
        # Keep track of the best model
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model
            best_alpha = alpha

    return best_model, best_alpha, best_auc

def process_split(folder_path, model_function, output_dir):
    """
    Process a single split from the specified folder.
    :param folder_path: Path to the folder containing train.csv, test.csv, and test_y.csv.
    :param model_function: Function to train and evaluate the model.
    :param output_dir: Directory to save the submission files.
    :return: AUC score for the split.
    """
    # File paths
    train_file = os.path.join(folder_path, 'train.csv')
    test_file = os.path.join(folder_path, 'test.csv')
    test_y_file = os.path.join(folder_path, 'test_y.csv')
    
    # Load data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    test_y = pd.read_csv(test_y_file)  # Only for AUC validation after predictions

    # Extract features and labels
    X_train = train_data.drop(columns=['id', 'sentiment', 'review'])
    y_train = train_data['sentiment']
    X_test = test_data.drop(columns=['id', 'review'])
    test_ids = test_data['id']
    y_test = test_y['sentiment']
    
    # Train the model
    model = model_function(X_train, y_train)
    
    # Predict probabilities for test data
    test_prob = model.predict_proba(X_test)[:, 1]
    
    # Save predictions to a CSV file
    split_name = os.path.basename(folder_path)
    submission_file = os.path.join(output_dir, f'mysubmission_{split_name}.csv')
    submission = pd.DataFrame({'id': test_ids, 'prob': test_prob})
    submission.to_csv(submission_file, index=False)
    print(f"{split_name}: Predictions saved to {submission_file}")
    
    # Compute AUC
    auc_score = roc_auc_score(y_test, test_prob)
    print(f"{split_name}: Test AUC = {auc_score:.3f}")
    return auc_score

def main_all_splits(output_dir):
    """
    Main pipeline to process all split folders.
    :param output_dir: Directory to save submission files.
    """
    current_dir = os.getcwd()  # Automatically get the current working directory
    print(f"Current working directory: {current_dir}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    total_auc = 0
    split_folders = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.startswith('split_')]
    
    for folder_path in split_folders:
        auc = process_split(folder_path, train_model, output_dir)
        total_auc += auc
    
    avg_auc = total_auc / len(split_folders)
    print(f"Average AUC across all splits: {avg_auc:.3f}")

def train_model(X_train, y_train):
    """
    Train a logistic regression model.
    :param X_train: Training features.
    :param y_train: Training labels.
    :return: Trained model.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

# # Execute the pipeline
# if __name__ == "__main__":
#     output_dir = 'submissions'  # Directory to save the submission files
#     main_all_splits(output_dir)



def load_bert_embeddings():

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def get_word_importance(review_text, embeddings, trained_model):

    words = review_text.split()
    
    feature_importance = trained_model.coef_[0]
    
    embedding_contributions = embeddings * feature_importance
    
    num_words = len(words)
    words_per_section = len(embeddings) // num_words
    
    word_scores = []
    for i in range(num_words):
        start_idx = i * words_per_section
        end_idx = start_idx + words_per_section if i < num_words - 1 else len(embeddings)
        section_contribution = np.mean(embedding_contributions[start_idx:end_idx])
        word_scores.append(section_contribution)
    
    return words, word_scores

def visualize_review_importance(words, scores, sentiment, save_path=None):
    scores = np.array(scores)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    html_text = []
    for word, score in zip(words, scores):
        if sentiment == 1:
            color = f'rgba(0, 0, 255, {score:.2f})'
        else:
            color = f'rgba(255, 0, 0, {score:.2f})'
            
        html_text.append(f'<span style="background-color: {color}">{word}</span> ')
    
    html = f'<p>{"".join(html_text)}</p>'
    
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html)
    
    return html

def analyze_reviews_interpretability(split_folder, trained_model, num_reviews=5):
    random.seed(51)
    
    test_data = pd.read_csv(os.path.join(split_folder, 'test.csv'))
    test_y = pd.read_csv(os.path.join(split_folder, 'test_y.csv'))
    test_data['sentiment'] = test_y['sentiment']
    
    pos_reviews = test_data[test_data['sentiment'] == 1].sample(num_reviews)
    neg_reviews = test_data[test_data['sentiment'] == 0].sample(num_reviews)
    selected_reviews = pd.concat([pos_reviews, neg_reviews])
    
    output_dir = os.path.join(split_folder, 'interpretability_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, row in selected_reviews.iterrows():
        embeddings = row.drop(['id', 'sentiment', 'review']).values
        
        review_text = row['review']
        
        words, scores = get_word_importance(review_text, embeddings, trained_model)
        
        html = visualize_review_importance(
            words,
            scores,
            row['sentiment'],
            save_path=os.path.join(output_dir, f'review_{idx}_visualization.html')
        )
        
        with open(os.path.join(output_dir, f'review_{idx}_metadata.txt'), 'w') as f:
            f.write(f"Review ID: {row['id']}\n")
            f.write(f"Sentiment: {'Positive' if row['sentiment'] == 1 else 'Negative'}\n")
            f.write(f"Original Review:\n{review_text}\n")

def main():
    output_dir = 'submissions'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    split1_folder = 'split_1'
    
    train_data = pd.read_csv(os.path.join(split1_folder, 'train.csv'))
    X_train = train_data.drop(columns=['id', 'sentiment', 'review'])
    y_train = train_data['sentiment']
    
    model = train_model(X_train, y_train)
    
    analyze_reviews_interpretability(split1_folder, model)
    
    main_all_splits(output_dir)

if __name__ == "__main__":
    main()