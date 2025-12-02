# Sistema de Recomendação - Resultados

## Método: Item-Item Collaborative Filtering

### Similaridade: Cosine Similarity

### Estatísticas do Dataset
- Total de reviews: 1,267,425
- Filmes únicos (coluna 'id'): 22,607
- Usuários/Críticos (coluna 'criticName'): 5,203
- Filmes no modelo (min 5 ratings): 22,607

### Métricas de Avaliação
- **RMSE (Root Mean Squared Error):** 0.4849
- **MAE (Mean Absolute Error):** 0.3717
- **MSE (Mean Squared Error):** 0.2351
- **Correlação (Pearson):** 0.8819
- **Predições realizadas:** 22,350
- **Usuários avaliados:** 500

### Interpretação
- RMSE < 1.0 em escala 1-5 indica boa precisão
- Correlação > 0.3 indica capacidade preditiva significativa

### Visualizações Geradas
Os gráficos foram salvos em: `/Users/rafapavao/Documents/WS/ia_2_ws/trab_3_rotten_tomatoes/movie-recomendation-system-ml/results/graphs`

- `confusion_matrix.png` - Matriz de confusão (ratings categorizados)
- `prediction_scatter.png` - Predições vs valores reais
- `error_distribution.png` - Distribuição dos erros de predição
- `similarity_heatmap.png` - Heatmap de similaridade entre filmes
- `rating_distribution.png` - Distribuição de ratings no dataset
- `sparsity_analysis.png` - Análise de esparsidade da matriz
- `top_movies.png` - Filmes mais avaliados e melhor avaliados
- `metrics_summary.png` - Resumo visual das métricas
- `dashboard_completo.png` - Dashboard consolidado
