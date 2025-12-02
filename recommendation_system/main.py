"""
Sistema de Recomendação de Filmes - Item-Item Collaborative Filtering
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import warnings
import os
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

MODEL_FILENAME = 'recommender_model.joblib'


def load_data(filepath: str) -> pd.DataFrame:
    """Carrega o dataset de reviews de filmes do Rotten Tomatoes."""
    print("Carregando dataset...")
    df = pd.read_csv(filepath)
    print(f"Dataset carregado: {len(df):,} reviews")
    return df


def standardize_score(score_str: str) -> float:
    """Padroniza os scores para uma escala de 0 a 5."""
    if pd.isna(score_str):
        return np.nan
    
    score_str = str(score_str).strip()
    
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
    
    letter_grades = {
        'A+': 5.0, 'A': 4.7, 'A-': 4.3,
        'B+': 4.0, 'B': 3.7, 'B-': 3.3,
        'C+': 3.0, 'C': 2.7, 'C-': 2.3,
        'D+': 2.0, 'D': 1.7, 'D-': 1.3,
        'F+': 1.0, 'F': 0.5, 'F-': 0.0
    }
    if score_str.upper() in letter_grades:
        return letter_grades[score_str.upper()]
    
    percent_match = re.match(r'^([\d.]+)\s*%$', score_str)
    if percent_match:
        return float(percent_match.group(1)) / 20
    
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
    """Cria um rating numérico a partir do score original ou do sentimento."""
    parsed_score = standardize_score(row.get('originalScore'))
    
    if not pd.isna(parsed_score):
        return max(1, min(5, parsed_score))
    
    sentiment = row.get('scoreSentiment', '')
    review_state = row.get('reviewState', '')
    
    if sentiment == 'POSITIVE' or review_state == 'fresh':
        return 4.0
    elif sentiment == 'NEGATIVE' or review_state == 'rotten':
        return 2.0
    else:
        return 3.0


def preprocess_data(df: pd.DataFrame, min_user_ratings: int = 10, min_movie_ratings: int = 10) -> pd.DataFrame:
    """Pré-processa os dados para o sistema de recomendação."""
    print("\nPré-processando dados...")
    
    df = df.copy()
    df = df.dropna(subset=['id', 'criticName'])
    df['rating'] = df.apply(create_rating_from_data, axis=1)
    
    if 'creationDate' in df.columns:
        df = df.sort_values('creationDate', ascending=False)
    df = df.drop_duplicates(subset=['criticName', 'id'], keep='first')
    
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)
        user_counts = df['criticName'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df['criticName'].isin(valid_users)]
        movie_counts = df['id'].value_counts()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
        df = df[df['id'].isin(valid_movies)]
    
    print(f"  Após pré-processamento: {len(df):,} reviews")
    print(f"    - Usuários: {df['criticName'].nunique():,}")
    print(f"    - Filmes: {df['id'].nunique():,}")
    
    return df


def create_ratings_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a matriz de ratings (usuários x filmes)."""
    ratings_matrix = df.pivot_table(
        index='criticName',
        columns='id',
        values='rating',
        aggfunc='mean'
    )
    return ratings_matrix


class ItemItemRecommender:
    """Sistema de Recomendação Item-Item Collaborative Filtering."""
    
    def __init__(self, min_ratings: int = 5, k_neighbors: int = 30):
        """Inicializa o recomendador."""
        self.min_ratings = min_ratings
        self.k_neighbors = k_neighbors
        self.ratings_matrix = None
        self.item_similarity = None
        self.movie_ids = None
        self.movie_titles = None
        self.user_ids = None
        self.global_mean = None
        self.user_bias = None
        self.item_bias = None
        
    def fit(self, ratings_matrix: pd.DataFrame):
        """Treina o modelo com a matriz de ratings."""
        print("\nTreinando modelo...")
        
        movie_rating_counts = ratings_matrix.notna().sum()
        valid_movies = movie_rating_counts[movie_rating_counts >= self.min_ratings].index
        self.ratings_matrix = ratings_matrix[valid_movies]
        
        self.movie_ids = list(self.ratings_matrix.columns)
        self.user_ids = list(self.ratings_matrix.index)
        
        self.movie_titles = {
            movie_id: movie_id.replace('_', ' ').replace('-', ' ').title() 
            for movie_id in self.movie_ids
        }
        
        all_ratings = self.ratings_matrix.stack()
        self.global_mean = all_ratings.mean()
        
        user_means = self.ratings_matrix.mean(axis=1)
        self.user_bias = user_means - self.global_mean
        
        item_means = self.ratings_matrix.mean(axis=0)
        self.item_bias = item_means - self.global_mean
        
        ratings_normalized = self.ratings_matrix.copy()
        for user in self.ratings_matrix.index:
            for movie in self.ratings_matrix.columns:
                if pd.notna(self.ratings_matrix.loc[user, movie]):
                    original = self.ratings_matrix.loc[user, movie]
                    normalized = original - self.global_mean - self.user_bias[user] - self.item_bias[movie]
                    ratings_normalized.loc[user, movie] = normalized
        
        ratings_filled = ratings_normalized.fillna(0)
        movie_features = ratings_filled.T.values
        similarity_matrix = cosine_similarity(movie_features)
        
        self.item_similarity = pd.DataFrame(
            similarity_matrix,
            index=self.movie_ids,
            columns=self.movie_ids
        )
        
        print(f"  Modelo treinado: {len(self.movie_ids)} filmes, {len(self.user_ids)} usuários")
        return self
    
    def get_similar_movies(self, movie_id: str, n: int = 10) -> pd.DataFrame:
        """Encontra os filmes mais similares a um filme dado."""
        if movie_id not in self.item_similarity.index:
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
        """Prediz o rating de um usuário para um filme usando k-NN com ajuste de bias."""
        if movie_id not in self.item_similarity.index:
            return np.nan
        
        baseline = self.global_mean + self.item_bias.get(movie_id, 0)
        
        if user_id is not None and user_id in self.user_bias.index:
            baseline += self.user_bias[user_id]
        
        similarities = []
        for rated_movie, rating in user_ratings.items():
            if rated_movie in self.item_similarity.index:
                sim = self.item_similarity.loc[movie_id, rated_movie]
                if sim > 0:
                    item_baseline = self.global_mean + self.item_bias.get(rated_movie, 0)
                    if user_id is not None and user_id in self.user_bias.index:
                        item_baseline += self.user_bias[user_id]
                    deviation = rating - item_baseline
                    similarities.append((sim, deviation))
        
        if not similarities:
            return np.clip(baseline, 1, 5)
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_k = similarities[:self.k_neighbors]
        
        weighted_sum = sum(sim * deviation for sim, deviation in top_k)
        similarity_sum = sum(sim for sim, _ in top_k)
        
        if similarity_sum > 0:
            adjustment = weighted_sum / similarity_sum
            prediction = baseline + adjustment
        else:
            prediction = baseline
        
        return np.clip(prediction, 1, 5)
    
    def recommend_for_user(self, user_ratings: dict, n: int = 10) -> pd.DataFrame:
        """Recomenda filmes para um usuário baseado nos seus ratings."""
        predictions = {}
        
        for movie_id in self.movie_ids:
            if movie_id in user_ratings:
                continue
            
            predicted = self.predict_rating(user_ratings, movie_id)
            if not np.isnan(predicted):
                predictions[movie_id] = predicted
        
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_predictions[:n]
        
        result = pd.DataFrame({
            'movie_id': [m[0] for m in top_n],
            'movie_title': [self.movie_titles.get(m[0], m[0]) for m in top_n],
            'predicted_rating': [m[1] for m in top_n]
        })
        
        return result
    
    def search_movies(self, query: str, limit: int = 20) -> list:
        """Busca filmes pelo nome."""
        query_lower = query.lower()
        results = []
        
        for movie_id in self.movie_ids:
            title = self.movie_titles.get(movie_id, movie_id)
            if query_lower in movie_id.lower() or query_lower in title.lower():
                results.append((movie_id, title))
        
        return results[:limit]
    
    def save(self, filepath: str):
        """Salva o modelo treinado em disco."""
        model_data = {
            'min_ratings': self.min_ratings,
            'k_neighbors': self.k_neighbors,
            'ratings_matrix': self.ratings_matrix,
            'item_similarity': self.item_similarity,
            'movie_ids': self.movie_ids,
            'movie_titles': self.movie_titles,
            'user_ids': self.user_ids,
            'global_mean': self.global_mean,
            'user_bias': self.user_bias,
            'item_bias': self.item_bias,
            'saved_at': datetime.now().isoformat()
        }
        joblib.dump(model_data, filepath)
        print(f"Modelo salvo em: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ItemItemRecommender':
        """Carrega um modelo salvo do disco."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
        
        model_data = joblib.load(filepath)
        
        recommender = cls(
            min_ratings=model_data['min_ratings'],
            k_neighbors=model_data['k_neighbors']
        )
        recommender.ratings_matrix = model_data['ratings_matrix']
        recommender.item_similarity = model_data['item_similarity']
        recommender.movie_ids = model_data['movie_ids']
        recommender.movie_titles = model_data['movie_titles']
        recommender.user_ids = model_data['user_ids']
        recommender.global_mean = model_data['global_mean']
        recommender.user_bias = model_data['user_bias']
        recommender.item_bias = model_data['item_bias']
        
        saved_at = model_data.get('saved_at', 'desconhecido')
        print(f"Modelo carregado de: {filepath}")
        print(f"  Salvo em: {saved_at}")
        print(f"  {len(recommender.movie_ids)} filmes, {len(recommender.user_ids)} usuários")
        
        return recommender


def get_model_path():
    """Retorna o caminho do arquivo do modelo."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, MODEL_FILENAME)


def train_model(save_model: bool = False):
    """Carrega dados e treina o modelo."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    DATA_PATH = os.path.join(project_root, 'rotten_tomatoes_movie_reviews.csv')
    MIN_RATINGS = 5
    K_NEIGHBORS = 30
    MIN_USER_RATINGS = 10
    MIN_MOVIE_RATINGS = 10
    
    print("Sistema de Recomendação de Filmes")
    print("Item-Item Collaborative Filtering\n")
    
    if not os.path.exists(DATA_PATH):
        print(f"Erro: Dataset não encontrado em {DATA_PATH}!")
        return None, None, None
    
    df = load_data(DATA_PATH)
    df = preprocess_data(df, min_user_ratings=MIN_USER_RATINGS, min_movie_ratings=MIN_MOVIE_RATINGS)
    
    ratings_matrix = create_ratings_matrix(df)
    
    recommender = ItemItemRecommender(min_ratings=MIN_RATINGS, k_neighbors=K_NEIGHBORS)
    recommender.fit(ratings_matrix)
    
    if save_model:
        recommender.save(get_model_path())
    
    return recommender, df, ratings_matrix


def load_saved_model():
    """Carrega o modelo salvo do disco."""
    model_path = get_model_path()
    
    print("Sistema de Recomendação de Filmes")
    print("Item-Item Collaborative Filtering\n")
    
    if not os.path.exists(model_path):
        print(f"Modelo salvo não encontrado em: {model_path}")
        print("Execute primeiro com --save para criar o modelo.")
        return None
    
    return ItemItemRecommender.load(model_path)


def main(use_saved: bool = False, save_after: bool = False):
    """Função principal para executar o sistema com métricas."""
    from metrics import calculate_rmse, create_visualizations, create_dashboard, save_results_report
    
    if use_saved:
        recommender = load_saved_model()
        df, ratings_matrix = None, None
        if recommender is not None:
            ratings_matrix = recommender.ratings_matrix
    else:
        recommender, df, ratings_matrix = train_model(save_model=save_after)
    
    if recommender is None:
        return None, None
    
    metrics = calculate_rmse(recommender, ratings_matrix, n_users=500)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'results')
    graphs_dir = os.path.join(results_dir, 'graphs')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    
    if df is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        DATA_PATH = os.path.join(project_root, 'rotten_tomatoes_movie_reviews.csv')
        df = load_data(DATA_PATH)
        df = preprocess_data(df)
    
    create_visualizations(recommender, metrics, df, ratings_matrix, graphs_dir)
    create_dashboard(recommender, metrics, df, graphs_dir)
    save_results_report(metrics, df, recommender, 5, results_dir, graphs_dir)
    
    print(f"\nResultados salvos em: {results_dir}")
    
    return recommender, metrics


def run_ui(use_saved: bool = False, save_after: bool = False):
    """Executa a interface Gradio."""
    from ui import run_interface
    
    if use_saved:
        recommender = load_saved_model()
    else:
        recommender, _, _ = train_model(save_model=save_after)
    
    if recommender is None:
        print("Erro ao inicializar o sistema.")
        return
    
    run_interface(recommender)


if __name__ == "__main__":
    import sys
    
    args = sys.argv[1:]
    use_saved = '--load' in args
    save_after = '--save' in args
    run_ui_mode = '--ui' in args
    
    if '--help' in args or '-h' in args:
        print("Sistema de Recomendação de Filmes")
        print("\nUso: python main.py [opções]")
        print("\nOpções:")
        print("  --ui      Inicia a interface web Gradio")
        print("  --save    Treina e salva o modelo para uso futuro")
        print("  --load    Carrega modelo salvo (pula treinamento)")
        print("  --help    Mostra esta mensagem de ajuda")
        print("\nExemplos:")
        print("  python main.py              # Treina e calcula métricas")
        print("  python main.py --save       # Treina, salva modelo e calcula métricas")
        print("  python main.py --load       # Carrega modelo salvo e calcula métricas")
        print("  python main.py --ui         # Treina e inicia interface web")
        print("  python main.py --ui --load  # Carrega modelo e inicia interface web")
        print("  python main.py --ui --save  # Treina, salva e inicia interface web")
        sys.exit(0)
    
    if run_ui_mode:
        run_ui(use_saved=use_saved, save_after=save_after)
    else:
        recommender, metrics = main(use_saved=use_saved, save_after=save_after)
        print("\nOpções de execução:")
        print("  python main.py --ui         # Interface web")
        print("  python main.py --save       # Salvar modelo")
        print("  python main.py --load       # Carregar modelo salvo")
        print("  python main.py --help       # Ajuda")
