"""
Movie Review Sentiment Analysis
===============================
This script trains classifiers to predict positive/negative sentiment from review text.
Uses the Movie Reviews Dataset (1.44M reviews) with scoreSentiment as the target variable.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.pipeline import Pipeline
import warnings
import re
import string
import joblib
import argparse

warnings.filterwarnings('ignore')


def load_data(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load the movie reviews dataset.
    
    Args:
        filepath: Path to the CSV file
        sample_size: Optional sample size for faster processing (None for full dataset)
    
    Returns:
        DataFrame with movie reviews
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"Sampled {len(df)} reviews for faster processing")
    
    print(f"Dataset loaded: {len(df):,} reviews")
    return df


def explore_data(df: pd.DataFrame) -> None:
    """
    Perform exploratory data analysis on the dataset.
    """
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*60)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    print("\nMissing Values:")
    print(df[['reviewText', 'scoreSentiment']].isnull().sum())
    
    print("\nSentiment Distribution:")
    print(df['scoreSentiment'].value_counts())
    print(f"\nSentiment Percentages:")
    print(df['scoreSentiment'].value_counts(normalize=True) * 100)
    
    df['text_length'] = df['reviewText'].astype(str).apply(len)
    df['word_count'] = df['reviewText'].astype(str).apply(lambda x: len(x.split()))
    
    print(f"\nReview Text Statistics:")
    print(f"  Average length (chars): {df['text_length'].mean():.0f}")
    print(f"  Average word count: {df['word_count'].mean():.0f}")
    print(f"  Min words: {df['word_count'].min()}")
    print(f"  Max words: {df['word_count'].max()}")


def visualize_data(df: pd.DataFrame) -> None:
    """
    Create visualizations for the dataset.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    sentiment_counts = df['scoreSentiment'].value_counts().sort_index()
    color_map = {'NEGATIVE': '#e74c3c', 'POSITIVE': '#2ecc71'}
    colors = [color_map[label] for label in sentiment_counts.index]
    sentiment_counts.plot(kind='bar', ax=ax1, color=colors)
    ax1.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=0)
    
    for i, v in enumerate(sentiment_counts):
        ax1.text(i, v + 1000, f'{v/len(df)*100:.1f}%', ha='center', fontweight='bold')
    
    ax2 = axes[0, 1]
    df['text_length'] = df['reviewText'].astype(str).apply(len)
    df.boxplot(column='text_length', by='scoreSentiment', ax=ax2)
    ax2.set_title('Review Length by Sentiment', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sentiment')
    ax2.set_ylabel('Character Count')
    plt.suptitle('')
    
    ax3 = axes[1, 0]
    df['word_count'] = df['reviewText'].astype(str).apply(lambda x: len(x.split()))
    for sentiment in df['scoreSentiment'].unique():
        subset = df[df['scoreSentiment'] == sentiment]['word_count']
        ax3.hist(subset, bins=50, alpha=0.6, label=sentiment)
    ax3.set_title('Word Count Distribution by Sentiment', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Word Count')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.set_xlim(0, 500)
    
    ax4 = axes[1, 1]
    critic_sentiment = df.groupby(['isTopCritic', 'scoreSentiment']).size().unstack(fill_value=0)
    if 'NEGATIVE' in critic_sentiment.columns and 'POSITIVE' in critic_sentiment.columns:
        critic_sentiment = critic_sentiment[['NEGATIVE', 'POSITIVE']]
    critic_colors = ['#e74c3c', '#2ecc71']
    critic_sentiment.plot(kind='bar', ax=ax4, color=critic_colors)
    ax4.set_title('Sentiment by Critic Type', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Is Top Critic')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=0)
    ax4.legend(title='Sentiment')
    
    plt.tight_layout()
    plt.savefig('data/sentiment_analysis_eda.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved to 'data/sentiment_analysis_eda.png'")


def clean_text(text: str) -> str:
    """
    Clean and preprocess review text.
    
    Args:
        text: Raw review text
    
    Returns:
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset for sentiment analysis.
    
    Args:
        df: Raw DataFrame
    
    Returns:
        Preprocessed DataFrame
    """
    print("\nPreprocessing data...")
    
    df_clean = df.dropna(subset=['reviewText', 'scoreSentiment']).copy()
    print(f"  Removed {len(df) - len(df_clean):,} rows with missing values")
    
    df_clean = df_clean[df_clean['scoreSentiment'].isin(['POSITIVE', 'NEGATIVE'])]
    df_clean['cleaned_text'] = df_clean['reviewText'].apply(clean_text)
    df_clean = df_clean[df_clean['cleaned_text'].str.len() > 0]
    df_clean['sentiment_label'] = (df_clean['scoreSentiment'] == 'POSITIVE').astype(int)
    
    print(f"  Final dataset size: {len(df_clean):,} reviews")
    print(f"  Positive reviews: {df_clean['sentiment_label'].sum():,} ({df_clean['sentiment_label'].mean()*100:.1f}%)")
    print(f"  Negative reviews: {(~df_clean['sentiment_label'].astype(bool)).sum():,} ({(1-df_clean['sentiment_label'].mean())*100:.1f}%)")
    
    return df_clean


def create_models() -> dict:
    """
    Create a dictionary of models to train.
    
    Returns:
        Dictionary of model pipelines
    """
    models = {
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.95,
                stop_words='english'
            )),
            ('clf', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
        ]),
        
        'Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.95,
                stop_words='english'
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ]),
        
        'Linear SVM': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.95,
                stop_words='english'
            )),
            ('clf', LinearSVC(random_state=42, max_iter=2000))
        ]),
        
        'Random Forest': Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=5,
                max_df=0.95,
                stop_words='english'
            )),
            ('clf', RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }
    
    return models


def train_and_evaluate(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    models: dict
) -> dict:
    """
    Train and evaluate all models.
    
    Args:
        X_train, X_test: Training and test review texts
        y_train, y_test: Training and test labels
        models: Dictionary of model pipelines
    
    Returns:
        Dictionary with evaluation results
    """
    results = {}
    
    print("\n" + "="*60)
    print("MODEL TRAINING AND EVALUATION")
    print("="*60)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_pred': y_pred
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
    
    return results


def plot_results(results: dict, y_test: pd.Series) -> None:
    """
    Create visualizations for model comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    ax1 = axes[0, 0]
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(results))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric] for model in results]
        ax1.bar(x + i*width, values, width, label=metric.capitalize())
    
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(results.keys(), rotation=15, ha='right')
    ax1.legend(loc='lower right')
    ax1.set_ylim(0.5, 1.0)
    ax1.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    ax2 = axes[0, 1]
    cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax2.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    ax3 = axes[1, 0]
    f1_scores = {name: results[name]['f1'] for name in results}
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(f1_scores)))
    bars = ax3.barh(list(f1_scores.keys()), list(f1_scores.values()), color=colors)
    ax3.set_xlabel('F1 Score')
    ax3.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlim(0.5, 1.0)
    
    for bar, score in zip(bars, f1_scores.values()):
        ax3.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', va='center', fontweight='bold')
    
    ax4 = axes[1, 1]
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    scatter = ax4.scatter(range(len(model_names)), accuracies, 
                         s=200, c=accuracies, cmap='RdYlGn', 
                         vmin=0.7, vmax=1.0, edgecolors='black')
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=15, ha='right')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Model Accuracy Overview', fontsize=14, fontweight='bold')
    ax4.set_ylim(0.7, 1.0)
    plt.colorbar(scatter, ax=ax4, label='Accuracy')
    
    plt.tight_layout()
    plt.savefig('data/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nModel comparison saved to 'data/model_comparison.png'")


def print_classification_reports(results: dict, y_test: pd.Series) -> None:
    """
    Print detailed classification reports for all models.
    """
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORTS")
    print("="*60)
    
    for name in results:
        print(f"\n{name}:")
        print("-" * 40)
        print(classification_report(
            y_test, 
            results[name]['y_pred'],
            target_names=['NEGATIVE', 'POSITIVE']
        ))


def show_feature_importance(results: dict, top_n: int = 20) -> None:
    """
    Show the most important features (words) for sentiment prediction.
    """
    print("\n" + "="*60)
    print("TOP FEATURES FOR SENTIMENT PREDICTION")
    print("="*60)
    
    if 'Logistic Regression' in results:
        model = results['Logistic Regression']['model']
        vectorizer = model.named_steps['tfidf']
        classifier = model.named_steps['clf']
        
        feature_names = vectorizer.get_feature_names_out()
        coefficients = classifier.coef_[0]
        
        top_positive_idx = np.argsort(coefficients)[-top_n:][::-1]
        top_negative_idx = np.argsort(coefficients)[:top_n]
        
        print(f"\nTop {top_n} words indicating POSITIVE sentiment:")
        print("-" * 40)
        for idx in top_positive_idx:
            print(f"  {feature_names[idx]:20} (coef: {coefficients[idx]:.4f})")
        
        print(f"\nTop {top_n} words indicating NEGATIVE sentiment:")
        print("-" * 40)
        for idx in top_negative_idx:
            print(f"  {feature_names[idx]:20} (coef: {coefficients[idx]:.4f})")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        ax1 = axes[0]
        pos_words = [feature_names[i] for i in top_positive_idx[:10]]
        pos_coefs = [coefficients[i] for i in top_positive_idx[:10]]
        ax1.barh(pos_words[::-1], pos_coefs[::-1], color='#2ecc71')
        ax1.set_xlabel('Coefficient')
        ax1.set_title('Top Words for POSITIVE Sentiment', fontsize=14, fontweight='bold')
        
        ax2 = axes[1]
        neg_words = [feature_names[i] for i in top_negative_idx[:10]]
        neg_coefs = [coefficients[i] for i in top_negative_idx[:10]]
        ax2.barh(neg_words[::-1], neg_coefs[::-1], color='#e74c3c')
        ax2.set_xlabel('Coefficient')
        ax2.set_title('Top Words for NEGATIVE Sentiment', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('data/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\nFeature importance saved to 'data/feature_importance.png'")


def predict_sentiment(model, text: str) -> dict:
    """
    Predict sentiment for a new review.
    
    Args:
        model: Trained model pipeline
        text: Review text to analyze
    
    Returns:
        Dictionary with prediction and probability
    """
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    
    if hasattr(model.named_steps['clf'], 'predict_proba'):
        proba = model.predict_proba([cleaned])[0]
        confidence = max(proba)
    else:
        confidence = None
    
    sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
    
    return {
        'text': text[:100] + '...' if len(text) > 100 else text,
        'sentiment': sentiment,
        'confidence': confidence
    }


def interactive_demo(model) -> None:
    """
    Run an interactive demo for sentiment prediction.
    """
    print("\n" + "="*60)
    print("INTERACTIVE SENTIMENT PREDICTION DEMO")
    print("="*60)
    
    sample_reviews = [
        "This movie was absolutely fantastic! The acting was superb and the story kept me engaged throughout.",
        "Terrible film. Waste of time and money. The plot made no sense and the acting was wooden.",
        "A decent movie with some good moments, but overall it felt mediocre and forgettable.",
        "One of the best films I've ever seen. A masterpiece of cinema that will stand the test of time.",
        "I couldn't even finish watching this garbage. Boring, predictable, and poorly executed."
    ]
    
    print("\nSample Predictions:")
    print("-" * 60)
    
    for review in sample_reviews:
        result = predict_sentiment(model, review)
        emoji = "✅" if result['sentiment'] == 'POSITIVE' else "❌"
        conf_str = f" (confidence: {result['confidence']:.2%})" if result['confidence'] else ""
        print(f"\n{emoji} {result['sentiment']}{conf_str}")
        print(f"   \"{result['text']}\"")


def save_best_model(results: dict, filepath: str = 'data/best_model.joblib') -> None:
    """
    Save the best performing model to disk.
    """
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    
    joblib.dump(best_model, filepath)
    print(f"\nBest model ({best_model_name}) saved to '{filepath}'")
    print(f"  F1 Score: {results[best_model_name]['f1']:.4f}")


def main():
    """
    Main function to run the complete sentiment analysis pipeline.
    """
    print("="*60)
    print("MOVIE REVIEW SENTIMENT ANALYSIS")
    print("="*60)
    
    DATA_PATH = 'data/dataset.csv'
    SAMPLE_SIZE = 100000
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    df = load_data(DATA_PATH, sample_size=SAMPLE_SIZE)
    explore_data(df)
    
    try:
        visualize_data(df)
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    df_clean = preprocess_data(df)
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean['cleaned_text'],
        df_clean['sentiment_label'],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_clean['sentiment_label']
    )
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set: {len(X_test):,} samples")
    
    models = create_models()
    results = train_and_evaluate(X_train, X_test, y_train, y_test, models)
    
    try:
        plot_results(results, y_test)
    except Exception as e:
        print(f"Plotting skipped: {e}")
    
    print_classification_reports(results, y_test)
    
    try:
        show_feature_importance(results)
    except Exception as e:
        print(f"Feature importance skipped: {e}")
    
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    interactive_demo(results[best_model_name]['model'])
    
    save_best_model(results)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\nBest Model: {best_model_name}")
    print(f"  Accuracy:  {results[best_model_name]['accuracy']:.4f}")
    print(f"  Precision: {results[best_model_name]['precision']:.4f}")
    print(f"  Recall:    {results[best_model_name]['recall']:.4f}")
    print(f"  F1-Score:  {results[best_model_name]['f1']:.4f}")
    print("\nPipeline completed successfully!")
    
    return results


def run_interactive_ui():
    """
    Launch an interactive Gradio UI for sentiment prediction.
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'gradio'])
        import gradio as gr
    
    model_path = 'data/best_model.joblib'
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from '{model_path}'")
    except FileNotFoundError:
        print(f"Model not found at '{model_path}'. Please run training first.")
        print("Run: python app.py --mode train")
        return
    
    def analyze_sentiment(review_text: str) -> tuple:
        """
        Analyze the sentiment of a review.
        """
        if not review_text or not review_text.strip():
            return "Please enter a review text.", "", None
        
        cleaned = clean_text(review_text)
        prediction = model.predict([cleaned])[0]
        
        if hasattr(model.named_steps['clf'], 'predict_proba'):
            proba = model.predict_proba([cleaned])[0]
            neg_prob = proba[0]
            pos_prob = proba[1]
        else:
            neg_prob = 1.0 if prediction == 0 else 0.0
            pos_prob = 1.0 if prediction == 1 else 0.0
        
        sentiment = "POSITIVE ✅" if prediction == 1 else "NEGATIVE ❌"
        confidence = max(pos_prob, neg_prob)
        
        result_text = f"""## Prediction: {sentiment}

**Confidence:** {confidence:.1%}

### Probability Breakdown:
- 🟢 Positive: {pos_prob:.1%}
- 🔴 Negative: {neg_prob:.1%}
"""
        
        confidence_data = {
            "POSITIVE 🟢": pos_prob,
            "NEGATIVE 🔴": neg_prob
        }
        
        return result_text, confidence_data
    
    examples = [
        ["This movie was absolutely fantastic! The acting was superb and the story kept me engaged throughout."],
        ["Terrible film. Waste of time and money. The plot made no sense and the acting was wooden."],
        ["A decent movie with some good moments, but overall it felt mediocre and forgettable."],
        ["One of the best films I've ever seen. A masterpiece of cinema that will stand the test of time."],
        ["I couldn't even finish watching this garbage. Boring, predictable, and poorly executed."],
        ["The cinematography was beautiful but the story was lacking depth."],
        ["An entertaining popcorn flick that doesn't take itself too seriously. Fun for the whole family!"],
        ["Disappointing sequel that fails to capture the magic of the original."]
    ]
    
    with gr.Blocks(title="Movie Review Sentiment Analyzer") as demo:
        gr.Markdown("""
        # 🎬 Movie Review Sentiment Analyzer
        
        Enter a movie review below and the AI will predict whether it's **POSITIVE** or **NEGATIVE**.
        
        This model was trained on 100,000+ movie critic reviews using Logistic Regression with TF-IDF features.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                review_input = gr.Textbox(
                    label="Enter your movie review",
                    placeholder="Type or paste a movie review here...",
                    lines=5,
                    max_lines=10
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Sentiment", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", scale=1)
            
            with gr.Column(scale=1):
                result_output = gr.Markdown(label="Result")
                confidence_output = gr.Label(
                    label="Confidence Scores",
                    num_top_classes=2
                )
        
        gr.Markdown("### 📝 Try these examples:")
        gr.Examples(
            examples=examples,
            inputs=review_input,
            outputs=[result_output, confidence_output],
            fn=analyze_sentiment,
            cache_examples=False
        )
        
        analyze_btn.click(
            fn=analyze_sentiment,
            inputs=review_input,
            outputs=[result_output, confidence_output]
        )
        
        review_input.submit(
            fn=analyze_sentiment,
            inputs=review_input,
            outputs=[result_output, confidence_output]
        )
        
        clear_btn.click(
            fn=lambda: ("", "", None),
            outputs=[review_input, result_output, confidence_output]
        )
        
        gr.Markdown("""
        ---
        ### About the Model
        - **Algorithm:** Logistic Regression with TF-IDF vectorization
        - **Features:** Unigrams and bigrams (1-2 word combinations)
        - **Training Data:** Movie critic reviews from various publications
        - **Performance:** ~85% F1-Score on test data
        """)
    
    print("\n" + "="*60)
    print("LAUNCHING INTERACTIVE UI")
    print("="*60)
    print("\nOpening in browser...")
    demo.launch(share=False, inbrowser=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Movie Review Sentiment Analysis')
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'ui', 'demo'],
        default='train',
        help='Mode to run: "train" to train models, "ui" for interactive UI, "demo" for command-line demo'
    )
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        results = main()
    elif args.mode == 'ui':
        run_interactive_ui()
    elif args.mode == 'demo':
        try:
            model = joblib.load('data/best_model.joblib')
            interactive_demo(model)
        except FileNotFoundError:
            print("Model not found. Please run training first: python app.py --mode train")
