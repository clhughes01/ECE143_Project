import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def load_and_clean_data(filepath):
    """
    Load and clean the Goodreads dataset with robust error handling
    """
    # Read the data with error handling for quoted fields
    df = pd.read_csv(
        filepath,
        escapechar='\\',
        quoting=1,  # QUOTE_ALL
        encoding='utf-8',
        on_bad_lines='skip'  # Skip problematic rows
    )
    
    # Convert publication_date to datetime with error handling
    df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
    
    # Clean missing values
    df = df.dropna(subset=['authors', 'average_rating'])
    
    # Clean authors field
    df['authors'] = df['authors'].astype(str)
    
    # Remove any potential duplicate books
    df = df.drop_duplicates(subset=['title', 'authors'])
    
    # Convert numeric columns
    numeric_columns = ['average_rating', 'num_pages', 'ratings_count', 'text_reviews_count']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Basic data validation
    print("\nDataset Summary:")
    print(f"Total number of books: {len(df)}")
    print(f"Number of unique authors: {df['authors'].nunique()}")
    print(f"Date range: {df['publication_date'].min()} to {df['publication_date'].max()}")
    
    return df

def create_author_metrics(df):
    """
    Create author-level metrics from book data
    """
    # Split multiple authors and explode the dataframe
    df['authors'] = df['authors'].str.split(',|&')  # Split on comma or ampersand
    df = df.explode('authors')
    df['authors'] = df['authors'].str.strip()
    
    # Filter out invalid author names
    df = df[df['authors'].str.len() > 1]  # Remove single-character authors
    
    # Calculate author-level metrics
    author_metrics = pd.DataFrame()
    
    # Basic metrics
    author_metrics['total_books'] = df.groupby('authors')['title'].count()
    author_metrics['avg_rating'] = df.groupby('authors')['average_rating'].mean()
    author_metrics['total_ratings'] = df.groupby('authors')['ratings_count'].sum()
    author_metrics['avg_reviews'] = df.groupby('authors')['text_reviews_count'].mean()
    
    # Calculate rating consistency (standard deviation of ratings)
    author_metrics['rating_consistency'] = df.groupby('authors')['average_rating'].std()
    
    # Calculate career span
    author_spans = df.groupby('authors').agg({
        'publication_date': ['min', 'max']
    })
    author_metrics['career_span_years'] = (
        author_spans['publication_date']['max'] - 
        author_spans['publication_date']['min']
    ).dt.days / 365.25
    
    # Fill NaN values in career span (for authors with single books)
    author_metrics['career_span_years'] = author_metrics['career_span_years'].fillna(0)
    
    # Calculate success score
    def normalize(series):
        if series.max() == series.min():
            return series - series.min()
        return (series - series.min()) / (series.max() - series.min())
    
    author_metrics['success_score'] = (
        normalize(author_metrics['avg_rating']) * 0.3 +
        normalize(np.log1p(author_metrics['total_ratings'])) * 0.3 +
        normalize(np.log1p(author_metrics['total_books'])) * 0.2 +
        normalize(1 / (author_metrics['rating_consistency'].fillna(1) + 1)) * 0.2
    )
    
    return author_metrics

def analyze_patterns(author_metrics):
    """
    Analyze patterns in author success
    """
    # Select only numeric columns for correlation
    numeric_cols = [
        'total_books', 'avg_rating', 'total_ratings', 
        'avg_reviews', 'rating_consistency', 
        'career_span_years', 'success_score'
    ]
    
    # Calculate correlations using only numeric columns
    correlations = author_metrics[numeric_cols].corr()
    
    # Segment authors based on success score
    author_metrics['success_tier'] = pd.qcut(
        author_metrics['success_score'], 
        q=5, 
        labels=['Bottom 20%', 'Lower Mid', 'Middle', 'Upper Mid', 'Top 20%']
    )
    
    # Calculate metrics by tier
    tier_metrics = author_metrics.groupby('success_tier')[numeric_cols].agg({
        'total_books': 'mean',
        'avg_rating': 'mean',
        'total_ratings': 'mean',
        'career_span_years': 'mean'
    }).round(2)
    
    print("\nMetrics by Success Tier:")
    print(tier_metrics)
    
    return correlations, author_metrics

def generate_insights(author_metrics):
    """
    Generate key insights from the analysis
    """
    # Select numeric columns for analysis
    numeric_cols = [
        'total_books', 'avg_rating', 'total_ratings', 
        'avg_reviews', 'rating_consistency', 
        'career_span_years', 'success_score'
    ]
    
    # Filter to authors with significant presence (more than 1 book)
    significant_authors = author_metrics[author_metrics['total_books'] > 1]
    
    # Calculate correlations only for numeric columns
    success_correlations = author_metrics[numeric_cols].corr()['success_score'].sort_values(ascending=False)
    
    # Get tier profiles without including the categorical column
    tier_profiles = author_metrics.groupby('success_tier')[numeric_cols].mean()
    
    insights = {
        'total_authors': len(author_metrics),
        'significant_authors': len(significant_authors),
        'top_authors': author_metrics.nlargest(10, 'success_score').index.tolist(),
        'avg_books_per_author': author_metrics['total_books'].mean(),
        'median_rating': author_metrics['avg_rating'].median(),
        'success_correlations': {
            factor: corr for factor, corr in success_correlations.items() 
            if factor != 'success_score'
        },
        'tier_profiles': tier_profiles.to_dict()
    }
    
    # Add percentile statistics
    insights['percentile_stats'] = {
        'top_10_percent_books': author_metrics['total_books'].quantile(0.9),
        'top_10_percent_ratings': author_metrics['total_ratings'].quantile(0.9),
        'top_10_percent_score': author_metrics['success_score'].quantile(0.9)
    }
    
    return insights

def create_visualizations(author_metrics, correlations):
    """
    Create visualizations for the analysis
    """
    # Create output directory if it doesn't exist
    os.makedirs('visualizations', exist_ok=True)
    
    # Set a default style
    plt.style.use('default')
    
    # Select numeric columns for visualizations
    numeric_cols = [
        'total_books', 'avg_rating', 'total_ratings', 
        'avg_reviews', 'rating_consistency', 
        'career_span_years', 'success_score'
    ]
    
    # 1. Success Score Distribution
    plt.figure(figsize=(12, 6))
    plt.hist(author_metrics['success_score'], bins=50, edgecolor='black')
    plt.title('Distribution of Author Success Scores', fontsize=12, pad=15)
    plt.xlabel('Success Score', fontsize=10)
    plt.ylabel('Count', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/success_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    im = plt.imshow(correlations, cmap='coolwarm', aspect='auto')
    plt.colorbar(im)
    
    # Add labels
    plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=45, ha='right')
    plt.yticks(range(len(numeric_cols)), numeric_cols)
    
    # Add correlation values
    for i in range(len(numeric_cols)):
        for j in range(len(numeric_cols)):
            text = plt.text(j, i, f'{correlations.iloc[i, j]:.2f}',
                          ha='center', va='center')
    
    plt.title('Correlation Matrix of Author Metrics', pad=15)
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Success Metrics by Tier
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['total_books', 'avg_rating', 'total_ratings', 'career_span_years']
    
    for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
        # Calculate mean values for each tier
        tier_means = author_metrics.groupby('success_tier')[metric].mean()
        
        # Create bar plot
        bars = ax.bar(range(len(tier_means)), tier_means.values)
        
        # Customize the plot
        ax.set_title(f'{metric.replace("_", " ").title()} by Success Tier')
        ax.set_xticks(range(len(tier_means)))
        ax.set_xticklabels(tier_means.index, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('visualizations/metrics_by_tier.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Success Factors Scatter Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['total_books', 'avg_rating', 'total_ratings', 'career_span_years']
    
    for ax, metric in zip(axes.flat, metrics):
        ax.scatter(author_metrics[metric], author_metrics['success_score'], 
                  alpha=0.5, s=20)
        ax.set_xlabel(metric.replace('_', ' ').title())
        ax.set_ylabel('Success Score')
        ax.grid(True, alpha=0.3)
        
        # Calculate and add correlation coefficient
        corr = author_metrics[metric].corr(author_metrics['success_score'])
        ax.set_title(f'{metric.replace("_", " ").title()} vs Success Score\n(correlation: {corr:.2f})')
    
    plt.tight_layout()
    plt.savefig('visualizations/success_factors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nVisualizations saved in 'visualizations' directory:")
    print("1. success_distribution.png - Shows the distribution of author success scores")
    print("2. correlation_heatmap.png - Shows correlations between different metrics")
    print("3. metrics_by_tier.png - Shows key metrics across success tiers")
    print("4. success_factors.png - Shows relationships between metrics and success score")

def main():
    try:
        # Load and process data
        print("Loading and cleaning data...")
        df = load_and_clean_data('data/books.csv')
        
        print("\nCreating author metrics...")
        author_metrics = create_author_metrics(df)
        
        print("\nAnalyzing patterns...")
        correlations, author_metrics = analyze_patterns(author_metrics)
        
        print("\nCreating visualizations...")
        create_visualizations(author_metrics, correlations)
        
        print("\nGenerating insights...")
        insights = generate_insights(author_metrics)
        
        # Print key findings
        print("\nKey Findings:")
        print(f"Total authors analyzed: {insights['total_authors']}")
        print(f"Authors with multiple books: {insights['significant_authors']}")
        
        print("\nTop 10 most successful authors:")
        for i, author in enumerate(insights['top_authors'], 1):
            print(f"{i}. {author}")
        
        print(f"\nAverage books per author: {insights['avg_books_per_author']:.2f}")
        print(f"Median author rating: {insights['median_rating']:.2f}")
        
        print("\nFactors most correlated with success:")
        for factor, corr in insights['success_correlations'].items():
            if factor != 'success_score':  # Skip self-correlation
                print(f"{factor}: {corr:.3f}")
        
        print("\nTop 10% threshold statistics:")
        for metric, value in insights['percentile_stats'].items():
            print(f"{metric.replace('_', ' ').title()}: {value:.2f}")
        
        # Print tier profiles
        print("\nMetrics by Success Tier:")
        tier_profiles = insights['tier_profiles']
        for metric, values in tier_profiles.items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            for tier, value in values.items():
                print(f"  {tier}: {value:.2f}")
                
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()