"""
Módulo de métricas e visualizações para o sistema de recomendação.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os


def calculate_rmse(recommender, ratings_matrix: pd.DataFrame, test_ratio: float = 0.2, n_users: int = 500) -> dict:
    """Calcula o RMSE do sistema de recomendação usando hold-out validation."""
    print("\nAvaliando modelo...")
    
    predictions = []
    actuals = []
    
    user_rating_counts = ratings_matrix.notna().sum(axis=1)
    eligible_users = user_rating_counts[user_rating_counts >= 5].index.tolist()
    
    np.random.seed(42)
    sample_size = min(n_users, len(eligible_users))
    sampled_users = np.random.choice(eligible_users, size=sample_size, replace=False)
    
    for i, user in enumerate(sampled_users):
        user_ratings = ratings_matrix.loc[user].dropna()
        
        n_holdout = max(1, int(len(user_ratings) * test_ratio))
        holdout_movies = np.random.choice(user_ratings.index, size=n_holdout, replace=False)
        
        train_ratings = {m: r for m, r in user_ratings.items() if m not in holdout_movies}
        
        if len(train_ratings) < 2:
            continue
        
        for movie in holdout_movies:
            if movie not in recommender.item_similarity.index:
                continue
            
            predicted = recommender.predict_rating(train_ratings, movie, user_id=user)
            
            if not np.isnan(predicted):
                predictions.append(predicted)
                actuals.append(user_ratings[movie])
    
    if not predictions:
        return {'error': 'Dados insuficientes para avaliação'}
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
    
    metrics = {
        'rmse': rmse, 'mse': mse, 'mae': mae, 'correlation': correlation,
        'n_predictions': len(predictions), 'n_users_evaluated': sample_size,
        'predictions': predictions, 'actuals': actuals
    }
    
    print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | Correlação: {correlation:.4f}")
    return metrics


def create_visualizations(recommender, metrics: dict, df: pd.DataFrame, ratings_matrix: pd.DataFrame, output_dir: str):
    """Cria visualizações e gráficos para análise do sistema de recomendação."""
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = metrics.get('predictions', np.array([]))
    actuals = metrics.get('actuals', np.array([]))
    
    _create_confusion_matrix(predictions, actuals, output_dir)
    _create_prediction_scatter(predictions, actuals, metrics, output_dir)
    _create_error_distribution(predictions, actuals, output_dir)
    _create_similarity_heatmap(recommender, output_dir)
    _create_rating_distribution(df, output_dir)
    _create_sparsity_analysis(ratings_matrix, output_dir)
    _create_top_movies_chart(df, output_dir)
    _create_metrics_summary(metrics, output_dir)


def _create_confusion_matrix(predictions: np.ndarray, actuals: np.ndarray, output_dir: str):
    """Cria matriz de confusão categorizando ratings."""
    def categorize(rating):
        if rating <= 2:
            return 'Baixo (1-2)'
        elif rating <= 3.5:
            return 'Médio (2.5-3.5)'
        else:
            return 'Alto (4-5)'
    
    pred_categories = [categorize(p) for p in predictions]
    actual_categories = [categorize(a) for a in actuals]
    
    categories = ['Baixo (1-2)', 'Médio (2.5-3.5)', 'Alto (4-5)']
    cm = confusion_matrix(actual_categories, pred_categories, labels=categories)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories, ax=ax)
    ax.set_xlabel('Rating Predito', fontsize=12)
    ax.set_ylabel('Rating Real', fontsize=12)
    ax.set_title('Matriz de Confusão\n(Ratings Categorizados)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _create_prediction_scatter(predictions: np.ndarray, actuals: np.ndarray, metrics: dict, output_dir: str):
    """Cria scatter plot de predições vs valores reais."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(actuals, predictions, alpha=0.3, edgecolors='none', s=30, c='steelblue')
    ax.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Predição Perfeita')
    
    z = np.polyfit(actuals, predictions, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(1, 5, 100)
    ax.plot(x_trend, p(x_trend), 'g-', linewidth=2, alpha=0.7, label=f'Tendência (r={metrics["correlation"]:.3f})')
    
    ax.set_xlabel('Rating Real', fontsize=12)
    ax.set_ylabel('Rating Predito', fontsize=12)
    ax.set_title(f'Predições vs Valores Reais\nRMSE: {metrics["rmse"]:.4f} | MAE: {metrics["mae"]:.4f}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.5, 5.5)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    textstr = f'n = {len(predictions):,}\nRMSE = {metrics["rmse"]:.4f}\nMAE = {metrics["mae"]:.4f}\nr = {metrics["correlation"]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _create_error_distribution(predictions: np.ndarray, actuals: np.ndarray, output_dir: str):
    """Cria distribuição dos erros de predição."""
    errors = predictions - actuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erro Zero')
    ax1.axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2, 
                label=f'Média: {np.mean(errors):.3f}')
    ax1.set_xlabel('Erro (Predito - Real)', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.set_title('Distribuição dos Erros de Predição', fontsize=14, fontweight='bold')
    ax1.legend()
    
    ax2 = axes[1]
    rating_bins = pd.cut(actuals, bins=[0, 2, 3, 4, 5], labels=['1-2', '2-3', '3-4', '4-5'])
    error_df = pd.DataFrame({'Erro': errors, 'Rating Real': rating_bins})
    
    error_df.boxplot(column='Erro', by='Rating Real', ax=ax2)
    ax2.set_xlabel('Faixa de Rating Real', fontsize=12)
    ax2.set_ylabel('Erro de Predição', fontsize=12)
    ax2.set_title('Erros por Faixa de Rating', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _create_similarity_heatmap(recommender, output_dir: str):
    """Cria heatmap da matriz de similaridade."""
    n_sample = min(30, len(recommender.movie_ids))
    
    similarity_variance = recommender.item_similarity.var(axis=1)
    top_variance_movies = similarity_variance.nlargest(n_sample).index.tolist()
    
    sample_similarity = recommender.item_similarity.loc[top_variance_movies, top_variance_movies]
    
    short_labels = [m[:20] + '...' if len(m) > 20 else m for m in sample_similarity.index]
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(sample_similarity, cmap='RdYlBu_r', center=0,
                xticklabels=short_labels, yticklabels=short_labels,
                square=True, ax=ax, cbar_kws={'label': 'Cosine Similarity'})
    
    ax.set_title(f'Heatmap de Similaridade entre Filmes\n(Amostra de {n_sample} filmes)', 
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _create_rating_distribution(df: pd.DataFrame, output_dir: str):
    """Cria distribuição de ratings no dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    df['rating'].hist(bins=20, ax=ax1, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=df['rating'].mean(), color='red', linestyle='--', linewidth=2,
                label=f'Média: {df["rating"].mean():.2f}')
    ax1.axvline(x=df['rating'].median(), color='green', linestyle='-', linewidth=2,
                label=f'Mediana: {df["rating"].median():.2f}')
    ax1.set_xlabel('Rating', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.set_title('Distribuição de Ratings no Dataset', fontsize=14, fontweight='bold')
    ax1.legend()
    
    ax2 = axes[1]
    rating_bins = pd.cut(df['rating'], bins=[0, 1, 2, 3, 4, 5], 
                         labels=['0-1', '1-2', '2-3', '3-4', '4-5'])
    rating_counts = rating_bins.value_counts().sort_index()
    
    bars = ax2.bar(rating_counts.index, rating_counts.values, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Faixa de Rating', fontsize=12)
    ax2.set_ylabel('Número de Reviews', fontsize=12)
    ax2.set_title('Contagem por Faixa de Rating', fontsize=14, fontweight='bold')
    
    for bar, count in zip(bars, rating_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _create_sparsity_analysis(ratings_matrix: pd.DataFrame, output_dir: str):
    """Analisa e visualiza a esparsidade da matriz."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ratings_per_user = ratings_matrix.notna().sum(axis=1)
    ax1.hist(ratings_per_user, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax1.axvline(x=ratings_per_user.mean(), color='blue', linestyle='--', linewidth=2,
                label=f'Média: {ratings_per_user.mean():.1f}')
    ax1.axvline(x=ratings_per_user.median(), color='green', linestyle='-', linewidth=2,
                label=f'Mediana: {ratings_per_user.median():.1f}')
    ax1.set_xlabel('Número de Ratings', fontsize=12)
    ax1.set_ylabel('Número de Usuários', fontsize=12)
    ax1.set_title('Ratings por Usuário (Crítico)', fontsize=14, fontweight='bold')
    ax1.legend()
    
    ax2 = axes[1]
    ratings_per_movie = ratings_matrix.notna().sum(axis=0)
    ax2.hist(ratings_per_movie, bins=50, edgecolor='black', alpha=0.7, color='mediumseagreen')
    ax2.axvline(x=ratings_per_movie.mean(), color='blue', linestyle='--', linewidth=2,
                label=f'Média: {ratings_per_movie.mean():.1f}')
    ax2.axvline(x=ratings_per_movie.median(), color='red', linestyle='-', linewidth=2,
                label=f'Mediana: {ratings_per_movie.median():.1f}')
    ax2.set_xlabel('Número de Ratings', fontsize=12)
    ax2.set_ylabel('Número de Filmes', fontsize=12)
    ax2.set_title('Ratings por Filme', fontsize=14, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sparsity_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _create_top_movies_chart(df: pd.DataFrame, output_dir: str):
    """Cria gráfico dos filmes mais avaliados e melhor avaliados."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    movie_counts = df.groupby('id').size().nlargest(15)
    movie_labels = [m[:25] + '...' if len(m) > 25 else m for m in movie_counts.index]
    
    bars1 = ax1.barh(range(len(movie_counts)), movie_counts.values, color='steelblue')
    ax1.set_yticks(range(len(movie_counts)))
    ax1.set_yticklabels(movie_labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Número de Reviews', fontsize=12)
    ax1.set_title('Top 15 Filmes Mais Avaliados', fontsize=14, fontweight='bold')
    
    for i, (bar, count) in enumerate(zip(bars1, movie_counts.values)):
        ax1.text(count + 10, i, f'{count:,}', va='center', fontsize=8)
    
    ax2 = axes[1]
    movie_stats = df.groupby('id').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['mean_rating', 'count']
    movie_stats = movie_stats[movie_stats['count'] >= 20]
    top_rated = movie_stats.nlargest(15, 'mean_rating')
    
    movie_labels2 = [m[:25] + '...' if len(m) > 25 else m for m in top_rated.index]
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_rated)))
    
    bars2 = ax2.barh(range(len(top_rated)), top_rated['mean_rating'].values, color=colors)
    ax2.set_yticks(range(len(top_rated)))
    ax2.set_yticklabels(movie_labels2, fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel('Rating Médio', fontsize=12)
    ax2.set_xlim(3.5, 5.1)
    ax2.set_title('Top 15 Filmes Melhor Avaliados\n(mín. 20 reviews)', fontsize=14, fontweight='bold')
    
    for i, (bar, rating) in enumerate(zip(bars2, top_rated['mean_rating'].values)):
        ax2.text(rating + 0.02, i, f'{rating:.2f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_movies.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _create_metrics_summary(metrics: dict, output_dir: str):
    """Cria um resumo visual das métricas."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    fig.suptitle('Resumo das Métricas de Avaliação\nSistema de Recomendação Item-Item', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    metrics_text = f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    MÉTRICAS DE DESEMPENHO                    ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║   RMSE (Root Mean Squared Error):     {metrics['rmse']:.4f}                ║
    ║   MSE  (Mean Squared Error):          {metrics['mse']:.4f}                ║
    ║   MAE  (Mean Absolute Error):         {metrics['mae']:.4f}                ║
    ║   Correlação (Pearson):               {metrics['correlation']:.4f}                ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                    DADOS DA AVALIAÇÃO                        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║   Predições Realizadas:               {metrics['n_predictions']:,}               ║
    ║   Usuários Avaliados:                 {metrics['n_users_evaluated']}                   ║
    ║                                                              ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                      INTERPRETAÇÃO                           ║
    ╠══════════════════════════════════════════════════════════════╣
    ║                                                              ║
    ║   • RMSE < 1.0 em escala 1-5: ✓ Boa precisão                ║
    ║   • Correlação > 0.3: ✓ Capacidade preditiva significativa  ║
    ║   • Método: Item-Item Collaborative Filtering               ║
    ║   • Similaridade: Cosine Similarity                         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, metrics_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_dashboard(recommender, metrics: dict, df: pd.DataFrame, output_dir: str):
    """Cria um dashboard consolidado com as principais métricas."""
    predictions = metrics.get('predictions', np.array([]))
    actuals = metrics.get('actuals', np.array([]))
    errors = predictions - actuals
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.scatter(actuals, predictions, alpha=0.3, edgecolors='none', s=20, c='steelblue')
    ax1.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Predição Perfeita')
    z = np.polyfit(actuals, predictions, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(1, 5, 100)
    ax1.plot(x_trend, p(x_trend), 'g-', linewidth=2, alpha=0.7, label=f'Tendência')
    ax1.set_xlabel('Rating Real', fontsize=10)
    ax1.set_ylabel('Rating Predito', fontsize=10)
    ax1.set_title(f'Predições vs Valores Reais (RMSE: {metrics["rmse"]:.4f})', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    metrics_text = f"""
    MÉTRICAS
    ─────────────────
    RMSE:  {metrics['rmse']:.4f}
    MSE:   {metrics['mse']:.4f}
    MAE:   {metrics['mae']:.4f}
    r:     {metrics['correlation']:.4f}
    ─────────────────
    Predições: {metrics['n_predictions']:,}
    Usuários:  {metrics['n_users_evaluated']}
    """
    ax2.text(0.5, 0.5, metrics_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax2.set_title('Resumo', fontsize=12, fontweight='bold')
    
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(errors, bins=40, edgecolor='black', alpha=0.7, color='coral')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2)
    ax3.set_xlabel('Erro (Predito - Real)', fontsize=10)
    ax3.set_ylabel('Frequência', fontsize=10)
    ax3.set_title('Distribuição dos Erros', fontsize=12, fontweight='bold')
    
    ax4 = fig.add_subplot(gs[1, 1])
    df['rating'].hist(bins=20, ax=ax4, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.axvline(x=df['rating'].mean(), color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Rating', fontsize=10)
    ax4.set_ylabel('Frequência', fontsize=10)
    ax4.set_title('Distribuição de Ratings no Dataset', fontsize=12, fontweight='bold')
    
    ax5 = fig.add_subplot(gs[1, 2])
    def categorize(rating):
        if rating <= 2:
            return 'Baixo'
        elif rating <= 3.5:
            return 'Médio'
        else:
            return 'Alto'
    pred_cat = [categorize(p) for p in predictions]
    actual_cat = [categorize(a) for a in actuals]
    categories = ['Baixo', 'Médio', 'Alto']
    cm = confusion_matrix(actual_cat, pred_cat, labels=categories)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=categories, yticklabels=categories, ax=ax5)
    ax5.set_xlabel('Predito', fontsize=10)
    ax5.set_ylabel('Real', fontsize=10)
    ax5.set_title('Matriz de Confusão', fontsize=12, fontweight='bold')
    
    ax6 = fig.add_subplot(gs[2, :2])
    n_sample = min(20, len(recommender.movie_ids))
    similarity_variance = recommender.item_similarity.var(axis=1)
    top_movies = similarity_variance.nlargest(n_sample).index.tolist()
    sample_sim = recommender.item_similarity.loc[top_movies, top_movies]
    short_labels = [m[:15] + '..' if len(m) > 15 else m for m in sample_sim.index]
    sns.heatmap(sample_sim, cmap='RdYlBu_r', center=0,
                xticklabels=short_labels, yticklabels=short_labels, ax=ax6)
    ax6.set_title(f'Heatmap de Similaridade (Amostra de {n_sample} filmes)', fontsize=12, fontweight='bold')
    plt.setp(ax6.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(ax6.get_yticklabels(), rotation=0, fontsize=7)
    
    ax7 = fig.add_subplot(gs[2, 2])
    movie_counts = df.groupby('id').size().nlargest(10)
    movie_labels = [m[:15] + '..' if len(m) > 15 else m for m in movie_counts.index]
    ax7.barh(range(len(movie_counts)), movie_counts.values, color='mediumseagreen')
    ax7.set_yticks(range(len(movie_counts)))
    ax7.set_yticklabels(movie_labels, fontsize=8)
    ax7.invert_yaxis()
    ax7.set_xlabel('Nº Reviews', fontsize=10)
    ax7.set_title('Top 10 Filmes\nMais Avaliados', fontsize=12, fontweight='bold')
    
    fig.suptitle('Dashboard - Sistema de Recomendação Item-Item Collaborative Filtering', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(output_dir, 'dashboard_completo.png'), dpi=150, bbox_inches='tight')
    plt.close()


def save_results_report(metrics: dict, df, recommender, min_ratings: int, results_dir: str, graphs_dir: str):
    """Salva relatório de resultados em Markdown."""
    with open(os.path.join(results_dir, 'recommendation_system_results.md'), 'w') as f:
        f.write("# Sistema de Recomendação - Resultados\n\n")
        f.write("## Método: Item-Item Collaborative Filtering\n\n")
        f.write("### Similaridade: Cosine Similarity\n\n")
        f.write("### Estatísticas do Dataset\n")
        f.write(f"- Total de reviews: {len(df):,}\n")
        f.write(f"- Filmes únicos (coluna 'id'): {df['id'].nunique():,}\n")
        f.write(f"- Usuários/Críticos (coluna 'criticName'): {df['criticName'].nunique():,}\n")
        f.write(f"- Filmes no modelo (min {min_ratings} ratings): {len(recommender.movie_ids):,}\n\n")
        f.write("### Métricas de Avaliação\n")
        f.write(f"- **RMSE (Root Mean Squared Error):** {metrics.get('rmse', 'N/A'):.4f}\n")
        f.write(f"- **MAE (Mean Absolute Error):** {metrics.get('mae', 'N/A'):.4f}\n")
        f.write(f"- **MSE (Mean Squared Error):** {metrics.get('mse', 'N/A'):.4f}\n")
        f.write(f"- **Correlação (Pearson):** {metrics.get('correlation', 'N/A'):.4f}\n")
        f.write(f"- **Predições realizadas:** {metrics.get('n_predictions', 'N/A'):,}\n")
        f.write(f"- **Usuários avaliados:** {metrics.get('n_users_evaluated', 'N/A')}\n\n")
        f.write("### Interpretação\n")
        f.write("- RMSE < 1.0 em escala 1-5 indica boa precisão\n")
        f.write("- Correlação > 0.3 indica capacidade preditiva significativa\n\n")
        f.write("### Visualizações Geradas\n")
        f.write(f"Os gráficos foram salvos em: `{graphs_dir}`\n\n")
        f.write("- `confusion_matrix.png` - Matriz de confusão (ratings categorizados)\n")
        f.write("- `prediction_scatter.png` - Predições vs valores reais\n")
        f.write("- `error_distribution.png` - Distribuição dos erros de predição\n")
        f.write("- `similarity_heatmap.png` - Heatmap de similaridade entre filmes\n")
        f.write("- `rating_distribution.png` - Distribuição de ratings no dataset\n")
        f.write("- `sparsity_analysis.png` - Análise de esparsidade da matriz\n")
        f.write("- `top_movies.png` - Filmes mais avaliados e melhor avaliados\n")
        f.write("- `metrics_summary.png` - Resumo visual das métricas\n")
        f.write("- `dashboard_completo.png` - Dashboard consolidado\n")
