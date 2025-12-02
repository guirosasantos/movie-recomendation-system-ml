"""
Módulo de interface gráfica (Gradio) para o sistema de recomendação.
"""

import gradio as gr


def create_gradio_interface(recommender):
    """Cria interface Gradio para o sistema de recomendação."""
    
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
                if movie_id in recommender.movie_ids:
                    user_ratings[movie_id] = rating
                    valid_movies.append((movie_id, rating))
                else:
                    matches = [m for m in recommender.movie_ids if movie_id.lower() in m.lower()]
                    if matches:
                        user_ratings[matches[0]] = rating
                        valid_movies.append((matches[0], rating))
        
        if len(user_ratings) == 0:
            return "❌ Por favor, informe pelo menos um filme válido.\n\nUse a busca para encontrar IDs de filmes."
        
        output = "## 📽️ Seus Filmes:\n\n"
        for movie_id, rating in valid_movies:
            title = recommender.movie_titles.get(movie_id, movie_id)
            stars = "⭐" * int(rating)
            output += f"• **{title}** - {rating:.1f} {stars}\n"
        
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


def run_interface(recommender):
    """Executa a interface Gradio."""
    print("\nIniciando Interface Web...")
    interface = create_gradio_interface(recommender)
    interface.launch(share=False)
