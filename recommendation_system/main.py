"""
Sistema de Recomendação de Filmes - Item-Item Collaborative Filtering
=====================================================================
Este módulo implementa um sistema de recomendação de filmes usando
Item-Item Collaborative Filtering com Cosine Similarity.

Dataset: Rotten Tomatoes Movie Reviews
- Coluna 'id': Nome dos filmes
- Coluna 'criticName': Nome dos usuários (críticos)
- Coluna 'originalScore': Score a ser padronizado

Métricas de Avaliação: RMSE (Root Mean Squared Error)
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import os
import gradio as gr

warnings.filterwarnings('ignore')

# Configuração de estilo para gráficos
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================================
# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO DOS DADOS
# ============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega o dataset de reviews de filmes do Rotten Tomatoes.
    
    Args:
        filepath: Caminho para o arquivo CSV
    
    Returns:
        DataFrame com as reviews
    """
    print("Carregando dataset...")
    df = pd.read_csv(filepath)
    
    print(f"Dataset carregado: {len(df):,} reviews")
    print(f"  - Filmes únicos (coluna 'id'): {df['id'].nunique():,}")
    print(f"  - Usuários/Críticos únicos (coluna 'criticName'): {df['criticName'].nunique():,}")
    
    return df


def standardize_score(score_str: str) -> float:
    """
    Padroniza os scores para uma escala de 0 a 5.
    
    Lida com diferentes formatos:
    - Frações: "3.5/4", "7/10", "85/100"
    - Letras: "A", "B+", "C-", etc.
    - Porcentagens: "85%"
    - Números puros
    
    Args:
        score_str: String do score original
    
    Returns:
        Score normalizado entre 0 e 5
    """
    if pd.isna(score_str):
        return np.nan
    
    score_str = str(score_str).strip()
    
    # Formato de fração (ex: "3.5/4", "7/10")
    fraction_match = re.match(r'^([\d.]+)\s*/\s*([\d.]+)$', score_str)
    if fraction_match:
        try:
            num_str = fraction_match.group(1).rstrip('.')
            denom_str = fraction_match.group(2).rstrip('.')
            numerator = float(num_str)
            denominator = float(denom_str)
            if denominator > 0:
                return (numerator / denominator) * 5
        except ValueError:
            pass
        return np.nan
    
    # Notas em letras
    letter_grades = {
        'A+': 5.0, 'A': 4.7, 'A-': 4.3,
        'B+': 4.0, 'B': 3.7, 'B-': 3.3,
        'C+': 3.0, 'C': 2.7, 'C-': 2.3,
        'D+': 2.0, 'D': 1.7, 'D-': 1.3,
        'F+': 1.0, 'F': 0.5, 'F-': 0.0
    }
    if score_str.upper() in letter_grades:
        return letter_grades[score_str.upper()]
    
    # Porcentagem (ex: "85%")
    percent_match = re.match(r'^([\d.]+)\s*%$', score_str)
    if percent_match:
        return float(percent_match.group(1)) / 20  # Converte para escala 0-5
    
    # Números puros
    try:
        num = float(score_str)
        if num <= 5:
            return num
        elif num <= 10:
            return num / 2
        elif num <= 100:
            return num / 20
    except ValueError:
        pass
    
    return np.nan


def create_rating_from_data(row: pd.Series) -> float:
    """
    Cria um rating numérico a partir do score original ou do sentimento.
    
    Args:
        row: Linha do DataFrame com originalScore e scoreSentiment
    
    Returns:
        Rating numérico entre 1 e 5
    """
    # Primeiro, tenta padronizar o score original
    parsed_score = standardize_score(row.get('originalScore'))
    
    if not pd.isna(parsed_score):
        # Garante que está no range 1-5
        return max(1, min(5, parsed_score))
    
    # Fallback para rating baseado em sentimento
    sentiment = row.get('scoreSentiment', '')
    review_state = row.get('reviewState', '')
    
    if sentiment == 'POSITIVE' or review_state == 'fresh':
        return 4.0
    elif sentiment == 'NEGATIVE' or review_state == 'rotten':
        return 2.0
    else:
        return 3.0


def preprocess_data(df: pd.DataFrame, min_user_ratings: int = 10, min_movie_ratings: int = 10) -> pd.DataFrame:
    """
    Pré-processa os dados para o sistema de recomendação.
    
    - Padroniza a coluna de score
    - Remove duplicatas
    - Limpa dados inválidos
    - Filtra usuários e filmes com poucos ratings
    
    Args:
        df: DataFrame original
        min_user_ratings: Mínimo de ratings por usuário para ser incluído
        min_movie_ratings: Mínimo de ratings por filme para ser incluído
    
    Returns:
        DataFrame pré-processado
    """
    print("\nPré-processando dados...")
    print(f"  Filtros: min {min_user_ratings} ratings/usuário, min {min_movie_ratings} ratings/filme")
    
    df = df.copy()
    
    # Remove linhas com filme ou crítico ausente
    df = df.dropna(subset=['id', 'criticName'])
    print(f"  Após remover valores nulos: {len(df):,} reviews")
    
    # Padroniza os scores
    print("  Padronizando scores para escala 0-5...")
    df['rating'] = df.apply(create_rating_from_data, axis=1)
    
    # Remove duplicatas (mesmo crítico avaliando mesmo filme)
    # Mantém a avaliação mais recente
    if 'creationDate' in df.columns:
        df = df.sort_values('creationDate', ascending=False)
    df = df.drop_duplicates(subset=['criticName', 'id'], keep='first')
    print(f"  Após remover duplicatas: {len(df):,} reviews")
    
    # Filtra usuários com poucos ratings (iterativo para convergência)
    print("  Filtrando usuários e filmes com poucos ratings...")
    prev_len = 0
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1
        
        # Filtra usuários
        user_counts = df['criticName'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df['criticName'].isin(valid_users)]
        
        # Filtra filmes
        movie_counts = df['id'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        df = df[df['id'].isin(valid_movies)]
    
    print(f"  Após filtrar (iteração {iteration}): {len(df):,} reviews")
    print(f"    - Usuários: {df['criticName'].nunique():,}")
    print(f"    - Filmes: {df['id'].nunique():,}")
    
    # Estatísticas dos ratings padronizados
    print(f"\n  Estatísticas dos ratings padronizados:")
    print(f"    Média: {df['rating'].mean():.2f}")
    print(f"    Desvio padrão: {df['rating'].std():.2f}")
    print(f"    Min: {df['rating'].min():.2f}, Max: {df['rating'].max():.2f}")
    
    return df


def create_ratings_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria a matriz de ratings (usuários x filmes).
    
    - Linhas: criticName (usuários)
    - Colunas: id (filmes)
    - Valores: rating padronizado
    
    Args:
        df: DataFrame pré-processado
    
    Returns:
        Matriz de ratings como DataFrame
    """
    print("\nCriando matriz de ratings...")
    
    ratings_matrix = df.pivot_table(
        index='criticName',  # Usuários nas linhas
        columns='id',        # Filmes nas colunas
        values='rating',
        aggfunc='mean'
    )
    
    print(f"  Forma da matriz: {ratings_matrix.shape}")
    print(f"  Usuários: {ratings_matrix.shape[0]:,}")
    print(f"  Filmes: {ratings_matrix.shape[1]:,}")
    
    sparsity = (ratings_matrix.isna().sum().sum() / ratings_matrix.size * 100)
    print(f"  Esparsidade: {sparsity:.2f}%")
    
    return ratings_matrix


# ============================================================================
# 2. ITEM-ITEM COLLABORATIVE FILTERING
# ============================================================================

class ItemItemRecommender:
    """
    Sistema de Recomendação Item-Item Collaborative Filtering.
    
    Utiliza Cosine Similarity para calcular a similaridade entre filmes
    baseado nos ratings dos usuários.
    
    Usa k-NN (k vizinhos mais próximos) para predições mais precisas.
    Inclui ajuste de bias (usuário e item) para melhor precisão.
    """
    
    def __init__(self, min_ratings: int = 5, k_neighbors: int = 30):
        """
        Inicializa o recomendador.
        
        Args:
            min_ratings: Número mínimo de ratings para incluir um filme
            k_neighbors: Número de vizinhos (k) para usar nas predições (k-NN)
        """
        self.min_ratings = min_ratings
        self.k_neighbors = k_neighbors
        self.ratings_matrix = None
        self.item_similarity = None
        self.movie_ids = None
        self.movie_titles = None
        self.user_ids = None
        
        # Bias terms
        self.global_mean = None
        self.user_bias = None
        self.item_bias = None
        
    def fit(self, ratings_matrix: pd.DataFrame):
        """
        Treina o modelo com a matriz de ratings.
        
        Calcula a matriz de similaridade item-item usando Cosine Similarity.
        Calcula os bias de usuário e item para ajuste nas predições.
        
        Args:
            ratings_matrix: Matriz de ratings (usuários x filmes)
        """
        print("\nTreinando modelo Item-Item Collaborative Filtering...")
        print(f"  Usando Cosine Similarity para matriz de similaridade...")
        print(f"  k-NN: k={self.k_neighbors} vizinhos para predições")
        print(f"  Ajuste de Bias: habilitado")
        
        # Filtra filmes com número mínimo de ratings
        movie_rating_counts = ratings_matrix.notna().sum()
        valid_movies = movie_rating_counts[movie_rating_counts >= self.min_ratings].index
        self.ratings_matrix = ratings_matrix[valid_movies]
        
        print(f"  Filmes com >= {self.min_ratings} ratings: {len(valid_movies):,}")
        
        # Armazena IDs de filmes e usuários
        self.movie_ids = list(self.ratings_matrix.columns)
        self.user_ids = list(self.ratings_matrix.index)
        
        # Cria títulos mais legíveis
        self.movie_titles = {
            movie_id: movie_id.replace('_', ' ').replace('-', ' ').title() 
            for movie_id in self.movie_ids
        }
        
        # ============================================================
        # CÁLCULO DOS BIAS (User Bias e Item Bias)
        # ============================================================
        print("  Calculando bias de usuários e itens...")
        
        # Média global (μ)
        all_ratings = self.ratings_matrix.stack()
        self.global_mean = all_ratings.mean()
        
        # Bias do usuário: b_u = média_usuário - μ
        user_means = self.ratings_matrix.mean(axis=1)
        self.user_bias = user_means - self.global_mean
        
        # Bias do item: b_i = média_item - μ
        item_means = self.ratings_matrix.mean(axis=0)
        self.item_bias = item_means - self.global_mean
        
        print(f"    Média global (μ): {self.global_mean:.3f}")
        print(f"    User bias range: [{self.user_bias.min():.3f}, {self.user_bias.max():.3f}]")
        print(f"    Item bias range: [{self.item_bias.min():.3f}, {self.item_bias.max():.3f}]")
        
        # ============================================================
        # NORMALIZAÇÃO DOS RATINGS (remove bias para similaridade)
        # ============================================================
        # Para calcular similaridade, usamos ratings normalizados
        # r_normalized = r - μ - b_u - b_i
        
        ratings_normalized = self.ratings_matrix.copy()
        for user in self.ratings_matrix.index:
            for movie in self.ratings_matrix.columns:
                if pd.notna(self.ratings_matrix.loc[user, movie]):
                    original = self.ratings_matrix.loc[user, movie]
                    normalized = original - self.global_mean - self.user_bias[user] - self.item_bias[movie]
                    ratings_normalized.loc[user, movie] = normalized
        
        # Preenche NaN com 0 para cálculo de similaridade
        ratings_filled = ratings_normalized.fillna(0)
        
        # Calcula similaridade item-item usando Cosine Similarity
        # Transpõe para que filmes sejam linhas
        movie_features = ratings_filled.T.values
        
        print("  Calculando matriz de similaridade (Cosine Similarity)...")
        similarity_matrix = cosine_similarity(movie_features)
        
        self.item_similarity = pd.DataFrame(
            similarity_matrix,
            index=self.movie_ids,
            columns=self.movie_ids
        )
        
        print(f"  Matriz de similaridade: {self.item_similarity.shape}")
        print("  Modelo treinado com sucesso!")
        
        return self
    
    def get_similar_movies(self, movie_id: str, n: int = 10) -> pd.DataFrame:
        """
        Encontra os filmes mais similares a um filme dado.
        
        Args:
            movie_id: ID do filme
            n: Número de filmes similares
        
        Returns:
            DataFrame com filmes similares e scores de similaridade
        """
        if movie_id not in self.item_similarity.index:
            # Tenta encontrar match parcial
            matches = [m for m in self.movie_ids if movie_id.lower() in m.lower()]
            if matches:
                movie_id = matches[0]
            else:
                return pd.DataFrame()
        
        similarities = self.item_similarity[movie_id].drop(movie_id)
        top_similar = similarities.nlargest(n)
        
        result = pd.DataFrame({
            'movie_id': top_similar.index,
            'movie_title': [self.movie_titles.get(m, m) for m in top_similar.index],
            'similarity_score': top_similar.values
        })
        
        return result
    
    def predict_rating(self, user_ratings: dict, movie_id: str, user_id: str = None) -> float:
        """
        Prediz o rating de um usuário para um filme usando k-NN com ajuste de bias.
        
        Fórmula: r_pred = μ + b_u + b_i + (soma ponderada dos desvios normalizados)
        
        Args:
            user_ratings: Dicionário {movie_id: rating}
            movie_id: ID do filme para predizer
            user_id: ID do usuário (opcional, para usar bias do usuário)
        
        Returns:
            Rating predito (entre 1 e 5)
        """
        if movie_id not in self.item_similarity.index:
            return np.nan
        
        # Baseline: μ + b_i (sem info do usuário, usa só bias do item)
        baseline = self.global_mean + self.item_bias.get(movie_id, 0)
        
        # Se temos o user_id, adiciona bias do usuário
        if user_id is not None and user_id in self.user_bias.index:
            baseline += self.user_bias[user_id]
        
        # Obtém similaridades com filmes que o usuário avaliou
        similarities = []
        for rated_movie, rating in user_ratings.items():
            if rated_movie in self.item_similarity.index:
                sim = self.item_similarity.loc[movie_id, rated_movie]
                if sim > 0:  # Apenas similaridades positivas
                    # Calcula o desvio normalizado do rating
                    # desvio = rating - (μ + b_i do filme avaliado)
                    item_baseline = self.global_mean + self.item_bias.get(rated_movie, 0)
                    if user_id is not None and user_id in self.user_bias.index:
                        item_baseline += self.user_bias[user_id]
                    deviation = rating - item_baseline
                    similarities.append((sim, deviation))
        
        if not similarities:
            # Retorna apenas o baseline se não houver similaridades
            return np.clip(baseline, 1, 5)
        
        # Ordena por similaridade e pega os k vizinhos mais próximos
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = similarities[:self.k_neighbors]
        
        # Calcula ajuste ponderado pela similaridade
        weighted_sum = sum(sim * deviation for sim, deviation in top_k)
        similarity_sum = sum(sim for sim, _ in top_k)
        
        if similarity_sum > 0:
            adjustment = weighted_sum / similarity_sum
            prediction = baseline + adjustment
        else:
            prediction = baseline
        
        # Garante que está no intervalo [1, 5]
        return np.clip(prediction, 1, 5)
    
    def recommend_for_user(self, user_ratings: dict, n: int = 10) -> pd.DataFrame:
        """
        Recomenda filmes para um usuário baseado nos seus ratings.
        
        Args:
            user_ratings: Dicionário {movie_id: rating}
            n: Número de recomendações
        
        Returns:
            DataFrame com filmes recomendados
        """
        predictions = {}
        
        for movie_id in self.movie_ids:
            if movie_id in user_ratings:
                continue
            
            predicted = self.predict_rating(user_ratings, movie_id)
            if not np.isnan(predicted):
                predictions[movie_id] = predicted
        
        # Ordena por rating predito
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_predictions[:n]
        
        result = pd.DataFrame({
            'movie_id': [m[0] for m in top_n],
            'movie_title': [self.movie_titles.get(m[0], m[0]) for m in top_n],
            'predicted_rating': [m[1] for m in top_n]
        })
        
        return result
    
    def search_movies(self, query: str, limit: int = 20) -> list:
        """
        Busca filmes pelo nome.
        
        Args:
            query: Termo de busca
            limit: Número máximo de resultados
        
        Returns:
            Lista de tuplas (movie_id, movie_title)
        """
        query_lower = query.lower()
        results = []
        
        for movie_id in self.movie_ids:
            title = self.movie_titles.get(movie_id, movie_id)
            if query_lower in movie_id.lower() or query_lower in title.lower():
                results.append((movie_id, title))
        
        return results[:limit]


# ============================================================================
# 3. AVALIAÇÃO COM RMSE
# ============================================================================

def calculate_rmse(recommender: ItemItemRecommender, 
                   ratings_matrix: pd.DataFrame,
                   test_ratio: float = 0.2,
                   n_users: int = 500) -> dict:
    """
    Calcula o RMSE do sistema de recomendação.
    
    Usa hold-out validation: para cada usuário, esconde parte dos ratings
    e tenta predizê-los.
    
    Args:
        recommender: Modelo treinado
        ratings_matrix: Matriz de ratings original
        test_ratio: Proporção de ratings para teste
        n_users: Número de usuários para avaliar
    
    Returns:
        Dicionário com métricas de avaliação
    """
    print("\n" + "=" * 60)
    print("AVALIAÇÃO DO MODELO - RMSE")
    print("=" * 60)
    
    predictions = []
    actuals = []
    
    # Seleciona usuários com ratings suficientes
    user_rating_counts = ratings_matrix.notna().sum(axis=1)
    eligible_users = user_rating_counts[user_rating_counts >= 5].index.tolist()
    
    np.random.seed(42)
    sample_size = min(n_users, len(eligible_users))
    sampled_users = np.random.choice(eligible_users, size=sample_size, replace=False)
    
    print(f"  Avaliando em {sample_size} usuários...")
    
    for i, user in enumerate(sampled_users):
        if (i + 1) % 100 == 0:
            print(f"    Processados {i + 1}/{sample_size} usuários...")
        
        user_ratings = ratings_matrix.loc[user].dropna()
        
        # Separa dados de treino e teste
        n_holdout = max(1, int(len(user_ratings) * test_ratio))
        holdout_movies = np.random.choice(user_ratings.index, size=n_holdout, replace=False)
        
        train_ratings = {m: r for m, r in user_ratings.items() if m not in holdout_movies}
        
        if len(train_ratings) < 2:
            continue
        
        # Prediz ratings dos filmes holdout
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
    
    # Calcula métricas
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    correlation = np.corrcoef(predictions, actuals)[0, 1] if len(predictions) > 1 else 0
    
    metrics = {
        'rmse': rmse,
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'n_predictions': len(predictions),
        'n_users_evaluated': sample_size,
        'predictions': predictions,
        'actuals': actuals
    }
    
    print("\n" + "-" * 60)
    print("RESULTADOS DA AVALIAÇÃO")
    print("-" * 60)
    print(f"  Predições realizadas: {len(predictions):,}")
    print(f"  Usuários avaliados: {sample_size}")
    print("-" * 60)
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  MSE  (Mean Squared Error):      {mse:.4f}")
    print(f"  MAE  (Mean Absolute Error):     {mae:.4f}")
    print(f"  Correlação:                     {correlation:.4f}")
    print("=" * 60)
    
    return metrics


# ============================================================================
# 3.1. VISUALIZAÇÕES E GRÁFICOS
# ============================================================================

def create_visualizations(recommender: ItemItemRecommender, 
                          metrics: dict, 
                          df: pd.DataFrame,
                          ratings_matrix: pd.DataFrame,
                          output_dir: str):
    """
    Cria visualizações e gráficos para análise do sistema de recomendação.
    
    Args:
        recommender: Modelo treinado
        metrics: Dicionário com métricas de avaliação
        df: DataFrame original pré-processado
        ratings_matrix: Matriz de ratings
        output_dir: Diretório para salvar os gráficos
    """
    print("\n" + "=" * 60)
    print("GERANDO VISUALIZAÇÕES")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    predictions = metrics.get('predictions', np.array([]))
    actuals = metrics.get('actuals', np.array([]))
    
    # 1. Matriz de Confusão (categorizando ratings)
    print("  1. Gerando Matriz de Confusão...")
    create_confusion_matrix(predictions, actuals, output_dir)
    
    # 2. Scatter Plot: Predito vs Real
    print("  2. Gerando Scatter Plot (Predito vs Real)...")
    create_prediction_scatter(predictions, actuals, metrics, output_dir)
    
    # 3. Distribuição dos Erros
    print("  3. Gerando Distribuição dos Erros...")
    create_error_distribution(predictions, actuals, output_dir)
    
    # 4. Heatmap de Similaridade (amostra de filmes)
    print("  4. Gerando Heatmap de Similaridade...")
    create_similarity_heatmap(recommender, output_dir)
    
    # 5. Distribuição de Ratings no Dataset
    print("  5. Gerando Distribuição de Ratings...")
    create_rating_distribution(df, output_dir)
    
    # 6. Análise de Esparsidade
    print("  6. Gerando Análise de Esparsidade...")
    create_sparsity_analysis(ratings_matrix, output_dir)
    
    # 7. Top Filmes por Número de Avaliações
    print("  7. Gerando Top Filmes...")
    create_top_movies_chart(df, output_dir)
    
    # 8. Métricas Resumo
    print("  8. Gerando Resumo de Métricas...")
    create_metrics_summary(metrics, output_dir)
    
    print(f"\n  ✓ Visualizações salvas em: {output_dir}")


def create_confusion_matrix(predictions: np.ndarray, actuals: np.ndarray, output_dir: str):
    """Cria matriz de confusão categorizando ratings."""
    
    # Categoriza ratings: Baixo (1-2), Médio (2.5-3.5), Alto (4-5)
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


def create_prediction_scatter(predictions: np.ndarray, actuals: np.ndarray, 
                              metrics: dict, output_dir: str):
    """Cria scatter plot de predições vs valores reais."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot com transparência
    ax.scatter(actuals, predictions, alpha=0.3, edgecolors='none', s=30, c='steelblue')
    
    # Linha de referência (predição perfeita)
    ax.plot([1, 5], [1, 5], 'r--', linewidth=2, label='Predição Perfeita')
    
    # Linha de tendência
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
    
    # Adiciona caixa de texto com estatísticas
    textstr = f'n = {len(predictions):,}\nRMSE = {metrics["rmse"]:.4f}\nMAE = {metrics["mae"]:.4f}\nr = {metrics["correlation"]:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_error_distribution(predictions: np.ndarray, actuals: np.ndarray, output_dir: str):
    """Cria distribuição dos erros de predição."""
    
    errors = predictions - actuals
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma dos erros
    ax1 = axes[0]
    ax1.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erro Zero')
    ax1.axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2, 
                label=f'Média: {np.mean(errors):.3f}')
    ax1.set_xlabel('Erro (Predito - Real)', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.set_title('Distribuição dos Erros de Predição', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # Box plot dos erros por faixa de rating real
    ax2 = axes[1]
    rating_bins = pd.cut(actuals, bins=[0, 2, 3, 4, 5], labels=['1-2', '2-3', '3-4', '4-5'])
    error_df = pd.DataFrame({'Erro': errors, 'Rating Real': rating_bins})
    
    error_df.boxplot(column='Erro', by='Rating Real', ax=ax2)
    ax2.set_xlabel('Faixa de Rating Real', fontsize=12)
    ax2.set_ylabel('Erro de Predição', fontsize=12)
    ax2.set_title('Erros por Faixa de Rating', fontsize=14, fontweight='bold')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
    plt.suptitle('')  # Remove título automático do boxplot
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_similarity_heatmap(recommender: ItemItemRecommender, output_dir: str):
    """Cria heatmap da matriz de similaridade (amostra de filmes)."""
    
    # Seleciona uma amostra de filmes para visualização
    n_sample = min(30, len(recommender.movie_ids))
    
    # Pega filmes com alta variância de similaridade (mais interessantes)
    similarity_variance = recommender.item_similarity.var(axis=1)
    top_variance_movies = similarity_variance.nlargest(n_sample).index.tolist()
    
    sample_similarity = recommender.item_similarity.loc[top_variance_movies, top_variance_movies]
    
    # Cria labels mais curtos
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


def create_rating_distribution(df: pd.DataFrame, output_dir: str):
    """Cria distribuição de ratings no dataset."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histograma de ratings
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
    
    # Contagem por faixa
    ax2 = axes[1]
    rating_bins = pd.cut(df['rating'], bins=[0, 1, 2, 3, 4, 5], 
                         labels=['0-1', '1-2', '2-3', '3-4', '4-5'])
    rating_counts = rating_bins.value_counts().sort_index()
    
    bars = ax2.bar(rating_counts.index, rating_counts.values, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Faixa de Rating', fontsize=12)
    ax2.set_ylabel('Número de Reviews', fontsize=12)
    ax2.set_title('Contagem por Faixa de Rating', fontsize=14, fontweight='bold')
    
    # Adiciona valores nas barras
    for bar, count in zip(bars, rating_counts.values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                f'{count:,}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_sparsity_analysis(ratings_matrix: pd.DataFrame, output_dir: str):
    """Analisa e visualiza a esparsidade da matriz."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Distribuição de ratings por usuário
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
    
    # Distribuição de ratings por filme
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


def create_top_movies_chart(df: pd.DataFrame, output_dir: str):
    """Cria gráfico dos filmes mais avaliados e melhor avaliados."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top 15 filmes mais avaliados
    ax1 = axes[0]
    movie_counts = df.groupby('id').size().nlargest(15)
    movie_labels = [m[:25] + '...' if len(m) > 25 else m for m in movie_counts.index]
    
    bars1 = ax1.barh(range(len(movie_counts)), movie_counts.values, color='steelblue')
    ax1.set_yticks(range(len(movie_counts)))
    ax1.set_yticklabels(movie_labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Número de Reviews', fontsize=12)
    ax1.set_title('Top 15 Filmes Mais Avaliados', fontsize=14, fontweight='bold')
    
    # Adiciona valores nas barras
    for i, (bar, count) in enumerate(zip(bars1, movie_counts.values)):
        ax1.text(count + 10, i, f'{count:,}', va='center', fontsize=8)
    
    # Top 15 filmes melhor avaliados (com mínimo de ratings)
    ax2 = axes[1]
    movie_stats = df.groupby('id').agg({'rating': ['mean', 'count']})
    movie_stats.columns = ['mean_rating', 'count']
    movie_stats = movie_stats[movie_stats['count'] >= 20]  # Mínimo 20 avaliações
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
    
    # Adiciona valores nas barras
    for i, (bar, rating) in enumerate(zip(bars2, top_rated['mean_rating'].values)):
        ax2.text(rating + 0.02, i, f'{rating:.2f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_movies.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_metrics_summary(metrics: dict, output_dir: str):
    """Cria um resumo visual das métricas."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Remove eixos
    ax.axis('off')
    
    # Título
    fig.suptitle('Resumo das Métricas de Avaliação\nSistema de Recomendação Item-Item', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Cria texto formatado
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


def create_all_in_one_dashboard(recommender: ItemItemRecommender,
                                 metrics: dict,
                                 df: pd.DataFrame,
                                 output_dir: str):
    """Cria um dashboard consolidado com as principais métricas."""
    
    print("  9. Gerando Dashboard Consolidado...")
    
    predictions = metrics.get('predictions', np.array([]))
    actuals = metrics.get('actuals', np.array([]))
    errors = predictions - actuals
    
    fig = plt.figure(figsize=(20, 16))
    
    # Layout do grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Scatter Plot (grande, canto superior esquerdo)
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
    
    # 2. Métricas Box (canto superior direito)
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
    
    # 3. Histograma de Erros
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(errors, bins=40, edgecolor='black', alpha=0.7, color='coral')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=np.mean(errors), color='green', linestyle='-', linewidth=2)
    ax3.set_xlabel('Erro (Predito - Real)', fontsize=10)
    ax3.set_ylabel('Frequência', fontsize=10)
    ax3.set_title('Distribuição dos Erros', fontsize=12, fontweight='bold')
    
    # 4. Distribuição de Ratings
    ax4 = fig.add_subplot(gs[1, 1])
    df['rating'].hist(bins=20, ax=ax4, edgecolor='black', alpha=0.7, color='steelblue')
    ax4.axvline(x=df['rating'].mean(), color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Rating', fontsize=10)
    ax4.set_ylabel('Frequência', fontsize=10)
    ax4.set_title('Distribuição de Ratings no Dataset', fontsize=12, fontweight='bold')
    
    # 5. Matriz de Confusão
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
    
    # 6. Heatmap de Similaridade (amostra)
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
    
    # 7. Top Filmes
    ax7 = fig.add_subplot(gs[2, 2])
    movie_counts = df.groupby('id').size().nlargest(10)
    movie_labels = [m[:15] + '..' if len(m) > 15 else m for m in movie_counts.index]
    ax7.barh(range(len(movie_counts)), movie_counts.values, color='mediumseagreen')
    ax7.set_yticks(range(len(movie_counts)))
    ax7.set_yticklabels(movie_labels, fontsize=8)
    ax7.invert_yaxis()
    ax7.set_xlabel('Nº Reviews', fontsize=10)
    ax7.set_title('Top 10 Filmes\nMais Avaliados', fontsize=12, fontweight='bold')
    
    # Título geral
    fig.suptitle('Dashboard - Sistema de Recomendação Item-Item Collaborative Filtering', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(os.path.join(output_dir, 'dashboard_completo.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Dashboard salvo em: {os.path.join(output_dir, 'dashboard_completo.png')}")

def create_gradio_interface(recommender: ItemItemRecommender):
    """
    Cria interface Gradio para o sistema de recomendação.
    
    Args:
        recommender: Modelo treinado
    
    Returns:
        Interface Gradio
    """
    
    def search_movies_ui(query: str) -> str:
        """Busca filmes pelo nome."""
        if not query or len(query) < 2:
            return "Digite pelo menos 2 caracteres para buscar..."
        
        results = recommender.search_movies(query, limit=30)
        
        if not results:
            return f"Nenhum filme encontrado para '{query}'"
        
        output = f"**Filmes encontrados ({len(results)}):**\n\n"
        for movie_id, title in results:
            output += f"• `{movie_id}`\n"
        
        return output
    
    def get_recommendations(movie1: str, rating1: float,
                           movie2: str, rating2: float,
                           movie3: str, rating3: float,
                           movie4: str, rating4: float,
                           movie5: str, rating5: float) -> str:
        """Gera recomendações baseadas nos filmes informados."""
        
        user_ratings = {}
        movies_input = [
            (movie1, rating1), (movie2, rating2), (movie3, rating3),
            (movie4, rating4), (movie5, rating5)
        ]
        
        valid_movies = []
        for movie_id, rating in movies_input:
            if movie_id and movie_id.strip():
                movie_id = movie_id.strip()
                # Verifica se o filme existe
                if movie_id in recommender.movie_ids:
                    user_ratings[movie_id] = rating
                    valid_movies.append((movie_id, rating))
                else:
                    # Tenta match parcial
                    matches = [m for m in recommender.movie_ids if movie_id.lower() in m.lower()]
                    if matches:
                        user_ratings[matches[0]] = rating
                        valid_movies.append((matches[0], rating))
        
        if len(user_ratings) == 0:
            return "❌ Por favor, informe pelo menos um filme válido.\n\nUse a busca para encontrar IDs de filmes."
        
        # Mostra filmes que o usuário informou
        output = "## 📽️ Seus Filmes:\n\n"
        for movie_id, rating in valid_movies:
            title = recommender.movie_titles.get(movie_id, movie_id)
            stars = "⭐" * int(rating)
            output += f"• **{title}** - {rating:.1f} {stars}\n"
        
        # Gera recomendações
        recommendations = recommender.recommend_for_user(user_ratings, n=10)
        
        if recommendations.empty:
            return output + "\n❌ Não foi possível gerar recomendações. Tente outros filmes."
        
        output += "\n## 🎬 Filmes Recomendados para Você:\n\n"
        
        for i, row in recommendations.iterrows():
            stars = "⭐" * int(round(row['predicted_rating']))
            output += f"**{i+1}. {row['movie_title']}**\n"
            output += f"   Rating predito: {row['predicted_rating']:.2f} {stars}\n\n"
        
        return output
    
    def get_similar_movies_ui(movie_id: str) -> str:
        """Encontra filmes similares."""
        if not movie_id or len(movie_id) < 2:
            return "Digite o ID de um filme..."
        
        movie_id = movie_id.strip()
        
        # Tenta match exato ou parcial
        if movie_id not in recommender.movie_ids:
            matches = [m for m in recommender.movie_ids if movie_id.lower() in m.lower()]
            if matches:
                movie_id = matches[0]
            else:
                return f"❌ Filme '{movie_id}' não encontrado. Use a busca para encontrar o ID correto."
        
        similar = recommender.get_similar_movies(movie_id, n=10)
        
        if similar.empty:
            return f"Nenhum filme similar encontrado para '{movie_id}'"
        
        title = recommender.movie_titles.get(movie_id, movie_id)
        output = f"## Filmes similares a **{title}**:\n\n"
        
        for i, row in similar.iterrows():
            output += f"**{i+1}. {row['movie_title']}**\n"
            output += f"   Similaridade: {row['similarity_score']:.3f}\n\n"
        
        return output
    
    # Cria interface
    with gr.Blocks(title="Sistema de Recomendação de Filmes") as interface:
        gr.Markdown("""
        # 🎬 Sistema de Recomendação de Filmes
        ### Item-Item Collaborative Filtering com Cosine Similarity
        
        Este sistema recomenda filmes baseado nos seus gostos pessoais.
        Informe alguns filmes que você gosta (ou não) e receba recomendações personalizadas!
        """)
        
        with gr.Tab("🔍 Buscar Filmes"):
            gr.Markdown("Busque filmes pelo nome para encontrar o ID correto:")
            search_input = gr.Textbox(label="Nome do Filme", placeholder="Ex: batman, matrix, star wars...")
            search_btn = gr.Button("Buscar", variant="primary")
            search_output = gr.Markdown()
            search_btn.click(search_movies_ui, inputs=search_input, outputs=search_output)
        
        with gr.Tab("⭐ Obter Recomendações"):
            gr.Markdown("""
            Informe até 5 filmes e suas notas (1-5) para receber recomendações personalizadas.
            Use a aba "Buscar Filmes" para encontrar os IDs dos filmes.
            """)
            
            with gr.Row():
                with gr.Column():
                    movie1 = gr.Textbox(label="Filme 1 (ID)", placeholder="ex: the_dark_knight")
                    rating1 = gr.Slider(1, 5, value=4, step=0.5, label="Nota")
                with gr.Column():
                    movie2 = gr.Textbox(label="Filme 2 (ID)", placeholder="ex: inception")
                    rating2 = gr.Slider(1, 5, value=4, step=0.5, label="Nota")
            
            with gr.Row():
                with gr.Column():
                    movie3 = gr.Textbox(label="Filme 3 (ID)", placeholder="ex: matrix")
                    rating3 = gr.Slider(1, 5, value=4, step=0.5, label="Nota")
                with gr.Column():
                    movie4 = gr.Textbox(label="Filme 4 (ID)", placeholder="opcional")
                    rating4 = gr.Slider(1, 5, value=4, step=0.5, label="Nota")
            
            with gr.Row():
                with gr.Column():
                    movie5 = gr.Textbox(label="Filme 5 (ID)", placeholder="opcional")
                    rating5 = gr.Slider(1, 5, value=4, step=0.5, label="Nota")
            
            recommend_btn = gr.Button("🎬 Gerar Recomendações", variant="primary", size="lg")
            recommendations_output = gr.Markdown()
            
            recommend_btn.click(
                get_recommendations,
                inputs=[movie1, rating1, movie2, rating2, movie3, rating3, movie4, rating4, movie5, rating5],
                outputs=recommendations_output
            )
        
        with gr.Tab("🎯 Filmes Similares"):
            gr.Markdown("Encontre filmes similares a um filme específico:")
            similar_input = gr.Textbox(label="ID do Filme", placeholder="ex: inception")
            similar_btn = gr.Button("Encontrar Similares", variant="primary")
            similar_output = gr.Markdown()
            similar_btn.click(get_similar_movies_ui, inputs=similar_input, outputs=similar_output)
        
        gr.Markdown("""
        ---
        **Sobre o Sistema:**
        - Método: Item-Item Collaborative Filtering
        - Similaridade: Cosine Similarity
        - Dataset: Rotten Tomatoes Movie Reviews
        """)
    
    return interface


# ============================================================================
# 5. FUNÇÃO PRINCIPAL
# ============================================================================

def main(calculate_metrics: bool = True):
    """Função principal para executar o sistema de recomendação.
    
    Args:
        calculate_metrics: Se True, calcula RMSE (mais lento). Default True.
    """
    
    # Configuração
    DATA_PATH = '../rotten_tomatoes_movie_reviews.csv'
    MIN_RATINGS = 5
    K_NEIGHBORS = 30  # Número de vizinhos para k-NN
    MIN_USER_RATINGS = 10  # Mínimo de ratings por usuário
    MIN_MOVIE_RATINGS = 10  # Mínimo de ratings por filme
    
    print("=" * 60)
    print("SISTEMA DE RECOMENDAÇÃO DE FILMES")
    print("Item-Item Collaborative Filtering com Cosine Similarity")
    print(f"k-NN: k={K_NEIGHBORS} vizinhos")
    print("=" * 60)
    
    # Verifica se o arquivo existe
    if not os.path.exists(DATA_PATH):
        DATA_PATH = 'rotten_tomatoes_movie_reviews.csv'
        if not os.path.exists(DATA_PATH):
            print(f"Erro: Dataset não encontrado!")
            return None, None
    
    # Carrega e pré-processa dados
    df = load_data(DATA_PATH)
    df = preprocess_data(df, min_user_ratings=MIN_USER_RATINGS, min_movie_ratings=MIN_MOVIE_RATINGS)
    
    # Cria matriz de ratings
    ratings_matrix = create_ratings_matrix(df)
    
    # Treina o modelo
    recommender = ItemItemRecommender(min_ratings=MIN_RATINGS, k_neighbors=K_NEIGHBORS)
    recommender.fit(ratings_matrix)
    
    # Calcula RMSE apenas se solicitado (não necessário para UI)
    metrics = None
    if calculate_metrics:
        metrics = calculate_rmse(recommender, ratings_matrix, n_users=500)
        
        # Salva resultados
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        results_dir = os.path.join(project_root, 'results')
        graphs_dir = os.path.join(results_dir, 'graphs')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(graphs_dir, exist_ok=True)
        
        # Gera visualizações
        create_visualizations(recommender, metrics, df, ratings_matrix, graphs_dir)
        create_all_in_one_dashboard(recommender, metrics, df, graphs_dir)
        
        with open(os.path.join(results_dir, 'recommendation_system_results.md'), 'w') as f:
            f.write("# Sistema de Recomendação - Resultados\n\n")
            f.write("## Método: Item-Item Collaborative Filtering\n\n")
            f.write("### Similaridade: Cosine Similarity\n\n")
            f.write("### Estatísticas do Dataset\n")
            f.write(f"- Total de reviews: {len(df):,}\n")
            f.write(f"- Filmes únicos (coluna 'id'): {df['id'].nunique():,}\n")
            f.write(f"- Usuários/Críticos (coluna 'criticName'): {df['criticName'].nunique():,}\n")
            f.write(f"- Filmes no modelo (min {MIN_RATINGS} ratings): {len(recommender.movie_ids):,}\n\n")
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
        
        print(f"\nResultados salvos em: {os.path.join(results_dir, 'recommendation_system_results.md')}")
        print(f"Gráficos salvos em: {graphs_dir}")
    
    return recommender, metrics


def run_ui():
    """Executa a interface Gradio."""
    # Não calcula RMSE no modo UI (não é necessário para recomendações)
    recommender, _ = main(calculate_metrics=False)
    
    if recommender is None:
        print("Erro ao inicializar o sistema.")
        return
    
    print("\n" + "=" * 60)
    print("Iniciando Interface Web...")
    print("=" * 60)
    
    interface = create_gradio_interface(recommender)
    interface.launch(share=False)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--ui':
        run_ui()
    else:
        # Modo avaliação: calcula RMSE
        recommender, metrics = main(calculate_metrics=True)
        print("\nPara iniciar a interface web, execute:")
        print("  python main.py --ui")
