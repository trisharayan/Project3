import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

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
    # current_dir = os.getcwd()  # Automatically get the current working directory
    # print(f"Current working directory: {current_dir}")
    
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    
    # total_auc = 0
    # split_folders = [os.path.join(current_dir, f) for f in os.listdir(current_dir) if f.startswith('split_')]
    
    # for folder_path in split_folders:
    #     auc = process_split(folder_path, train_model, output_dir)
    #     total_auc += auc
    
    # avg_auc = total_auc / len(split_folders)
    # print(f"Average AUC across all splits: {avg_auc:.3f}")

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
    
    split_1_path = [f for f in split_folders if f.endswith('split_1')][0]
    test_file = os.path.join(split_1_path, 'test.csv')
    trained_model_path = 'path_to_trained_model'
    
    print("\nRunning interpretability analysis for split_1...")
    interpretability_analysis(test_file, trained_model_path)

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

import os
import random
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
import torch

random.seed(42)

def interpretability_analysis(test_file, trained_model_path):
    script_directory = os.getcwd()
    
    test_data = pd.read_csv(test_file)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(trained_model_path)
    model.eval()

    positive_reviews = test_data[test_data['sentiment'] == 1]['review']
    negative_reviews = test_data[test_data['sentiment'] == 0]['review']

    selected_positive = positive_reviews.sample(5, random_state=42)
    selected_negative = negative_reviews.sample(5, random_state=42)
    selected_reviews = pd.concat([selected_positive, selected_negative])

    highlighted_reviews = []
    for review in selected_reviews:
        inputs = tokenizer(review, return_tensors="pt", truncation=True, max_length=512)
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        
        tokens = tokenizer.tokenize(review)
        token_probs = probs.detach().numpy()[0]
        important_tokens = [(token, token_probs[i]) for i, token in enumerate(tokens)]
        important_tokens.sort(key=lambda x: x[1], reverse=True)
        highlighted_reviews.append((review, important_tokens[:5]))

    for i, (review, important_tokens) in enumerate(highlighted_reviews):
        print(f"Review {i+1} (Prediction: {'Positive' if prediction else 'Negative'}):")
        print("Original Review:", review)
        print("Highlighted Tokens:", [token for token, _ in important_tokens])
        print()

    for i, (review, important_tokens) in enumerate(highlighted_reviews):
        fig, ax = plt.subplots(figsize=(10, 6))
        words, scores = zip(*important_tokens)
        ax.barh(words, scores, color='skyblue')
        ax.set_title(f"Review {i+1}: Key Words Impacting Sentiment")
        ax.set_xlabel("Importance Score")
        ax.invert_yaxis()
        plt.tight_layout()
        save_path = os.path.join(script_directory, f"review_{i+1}_interpretation.png")
        plt.savefig(save_path)
        print(f"Saved visualization for Review {i+1} at {save_path}")
        plt.close()


# Execute the pipeline
if __name__ == "__main__":
    output_dir = 'submissions'  # Directory to save the submission files
    main_all_splits(output_dir)
