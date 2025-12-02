# 🎬 Sistema de Recomendação de Filmes - Documentação

## Item-Item Collaborative Filtering com Cosine Similarity

---

## Índice

1. [Visão Geral](#1-visão-geral)
2. [Como Executar](#2-como-executar)
3. [Dataset](#3-dataset)
4. [Pré-processamento dos Dados](#4-pré-processamento-dos-dados)
5. [Sistema de Recomendação](#5-sistema-de-recomendação)
6. [Técnicas de Otimização](#6-técnicas-de-otimização)
7. [Avaliação do Modelo](#7-avaliação-do-modelo)
8. [Resultados](#8-resultados)
9. [Fluxo Completo](#9-fluxo-completo)

---

## 1. Visão Geral

Este projeto implementa um **Sistema de Recomendação de Filmes** utilizando a técnica de **Item-Item Collaborative Filtering** com **Cosine Similarity**.

### Objetivo

Dado um conjunto de avaliações de filmes feitas por críticos, o sistema é capaz de:
- Predizer a nota que um usuário daria para um filme que ele ainda não avaliou
- Recomendar filmes similares a um filme específico
- Gerar recomendações personalizadas baseadas no histórico do usuário

### Tecnologias Utilizadas

| Biblioteca | Uso |
|------------|-----|
| **Pandas** | Manipulação de dados e DataFrames |
| **NumPy** | Operações numéricas e matriciais |
| **Scikit-learn** | Cálculo de Cosine Similarity |
| **Matplotlib/Seaborn** | Visualizações e gráficos |
| **Gradio** | Interface web interativa |
| **Joblib** | Persistência do modelo treinado |

### Tipo de Machine Learning

| Aspecto | Descrição |
|---------|-----------|
| **Tipo** | Aprendizado Não Supervisionado |
| **Algoritmo** | Item-Item Collaborative Filtering |
| **Métrica de Similaridade** | Cosine Similarity |
| **Otimizações** | k-NN + User/Item Bias |

---

## 2. Como Executar

### Estrutura de Arquivos

```
recommendation_system/
├── main.py       # Lógica principal e classe ItemItemRecommender
├── metrics.py    # Cálculo de métricas e visualizações
├── ui.py         # Interface Gradio
└── DOCUMENTATION.md
```

### Instalação de Dependências

```bash
pip install pandas numpy scikit-learn matplotlib seaborn gradio joblib
```

### Modos de Execução

| Comando | Descrição |
|---------|-----------|
| `python main.py` | Treina o modelo e calcula métricas (RMSE, MAE, etc.) |
| `python main.py --save` | Treina, **salva o modelo** e calcula métricas |
| `python main.py --load` | **Carrega modelo salvo** e calcula métricas (pula treinamento) |
| `python main.py --ui` | Treina o modelo e inicia a interface web Gradio |
| `python main.py --ui --save` | Treina, salva o modelo e inicia a interface web |
| `python main.py --ui --load` | Carrega modelo salvo e inicia a interface web |
| `python main.py --help` | Mostra ajuda com todas as opções |

### Exemplos de Uso

```bash
# Primeira execução: treinar e salvar o modelo
python main.py --save

# Execuções posteriores: carregar modelo salvo (muito mais rápido!)
python main.py --load

# Iniciar interface web com modelo salvo
python main.py --ui --load

# Ver todas as opções
python main.py --help
```

### Persistência do Modelo

O modelo treinado é salvo em `data/recommender_model.joblib` e contém:
- Matriz de similaridade entre filmes
- Bias de usuários e itens
- Metadados (IDs de filmes, títulos, etc.)
- Data/hora do salvamento

**Vantagem**: Carregar um modelo salvo leva ~2 segundos, enquanto treinar do zero leva ~2 minutos.

---

## 3. Dataset

### Fonte
**Rotten Tomatoes Movie Reviews Dataset**

### Estrutura das Colunas

| Coluna | Descrição | Uso no Sistema |
|--------|-----------|----------------|
| `id` | Identificador único do filme | **Item** (filme) |
| `criticName` | Nome do crítico | **Usuário** |
| `originalScore` | Nota original (vários formatos) | **Rating** (após padronização) |
| `scoreSentiment` | Sentimento (POSITIVE/NEGATIVE) | Fallback para rating |
| `reviewState` | Estado da review (fresh/rotten) | Fallback para rating |
| `creationDate` | Data da avaliação | Ordenação para duplicatas |

### Estatísticas do Dataset Original

```
Total de reviews: 1,444,963
Filmes únicos: 69,263
Críticos únicos: 15,510
```

---

## 4. Pré-processamento dos Dados

### 4.1 Padronização dos Scores

O dataset contém scores em diversos formatos que precisam ser normalizados para uma escala uniforme de **1 a 5**.

#### Formatos Tratados

| Formato Original | Exemplo | Fórmula de Conversão |
|------------------|---------|----------------------|
| Fração | "3.5/4", "7/10", "85/100" | `(numerador / denominador) × 5` |
| Letra | "A+", "B-", "C" | Mapeamento direto para valor numérico |
| Porcentagem | "85%" | `valor / 20` |
| Número (escala 10) | "8" | `valor / 2` |
| Número (escala 100) | "75" | `valor / 20` |

#### Função de Padronização

```python
def standardize_score(score_str: str) -> float:
    # 1. Formato de fração (ex: "3.5/4", "7/10")
    fraction_match = re.match(r'^([\d.]+)\s*/\s*([\d.]+)$', score_str)
    if fraction_match:
        numerator = float(fraction_match.group(1))
        denominator = float(fraction_match.group(2))
        return (numerator / denominator) * 5
    
    # 2. Notas em letras
    letter_grades = {
        'A+': 5.0, 'A': 4.7, 'A-': 4.3,
        'B+': 4.0, 'B': 3.7, 'B-': 3.3,
        'C+': 3.0, 'C': 2.7, 'C-': 2.3,
        'D+': 2.0, 'D': 1.7, 'D-': 1.3,
        'F+': 1.0, 'F': 0.5, 'F-': 0.0
    }
    
    # 3. Porcentagem (ex: "85%")
    # 4. Números puros (detecta escala automaticamente)
```

#### Fallback para Sentimento

Quando o `originalScore` não pode ser interpretado, utiliza-se o sentimento:

```python
if sentiment == 'POSITIVE' or review_state == 'fresh':
    return 4.0
elif sentiment == 'NEGATIVE' or review_state == 'rotten':
    return 2.0
else:
    return 3.0  # Neutro
```

### 4.2 Filtragem de Qualidade

Para melhorar a qualidade das recomendações, aplicamos **filtros iterativos** que removem usuários e filmes com poucos ratings.

#### Parâmetros de Filtragem

```python
MIN_USER_RATINGS = 10   # Mínimo de avaliações por usuário
MIN_MOVIE_RATINGS = 10  # Mínimo de avaliações por filme
```

#### Por que Filtragem Iterativa?

O processo é iterativo porque remover usuários pode fazer filmes ficarem abaixo do mínimo (e vice-versa):

```python
while len(df) != prev_len:
    prev_len = len(df)
    
    # Filtra usuários com poucos ratings
    user_counts = df['criticName'].value_counts()
    valid_users = user_counts[user_counts >= min_user_ratings].index
    df = df[df['criticName'].isin(valid_users)]
    
    # Filtra filmes com poucos ratings
    movie_counts = df['id'].value_counts()
    valid_movies = movie_counts[movie_counts >= min_movie_ratings].index
    df = df[df['id'].isin(valid_movies)]
```

#### Resultado da Filtragem

| Métrica | Antes | Depois | Redução |
|---------|-------|--------|---------|
| Reviews | 1,426,442 | 1,267,425 | 11.1% |
| Usuários | 15,510 | 5,203 | 66.5% |
| Filmes | 69,263 | 22,607 | 67.4% |
| Esparsidade | 99.87% | 98.92% | ↓ melhor |

### 3.3 Criação da Matriz de Ratings

A **Matriz User-Item** é a estrutura central do sistema:

```python
ratings_matrix = df.pivot_table(
    index='criticName',  # Linhas: Usuários
    columns='id',        # Colunas: Filmes
    values='rating',     # Valores: Ratings padronizados
    aggfunc='mean'       # Média em caso de duplicatas
)
```

#### Estrutura da Matriz

```
                    filme_1  filme_2  filme_3  ...  filme_22607
usuario_1             4.0      NaN      3.5   ...      NaN
usuario_2             NaN      5.0      NaN   ...      2.0
usuario_3             3.0      4.5      4.0   ...      NaN
...
usuario_5203          NaN      NaN      5.0   ...      4.0
```

- **Dimensão**: 5,203 usuários × 22,607 filmes
- **Esparsidade**: 98.92% (a maioria das células é NaN)

---

## 5. Sistema de Recomendação

### 4.1 Item-Item Collaborative Filtering

#### Conceito

O Item-Item Collaborative Filtering baseia-se na ideia de que **filmes similares tendem a receber avaliações similares**. 

Em vez de encontrar usuários similares (User-User), encontramos **itens (filmes) similares** baseado no padrão de avaliações que receberam.

#### Vantagens do Item-Item sobre User-User

| Aspecto | Item-Item | User-User |
|---------|-----------|-----------|
| **Escalabilidade** | ✅ Nº de itens é menor e estável | ❌ Nº de usuários cresce muito |
| **Estabilidade** | ✅ Similaridades mudam pouco | ❌ Novos usuários alteram tudo |
| **Cold Start** | ✅ Novos usuários ok | ❌ Novos usuários problemáticos |
| **Interpretabilidade** | ✅ "Você gostou de X, vai gostar de Y" | ❌ Menos intuitivo |

### 4.2 Cosine Similarity

#### Definição

A similaridade entre dois filmes A e B é calculada como o **cosseno do ângulo** entre seus vetores de ratings:

$$\text{similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}$$

Expandindo:

$$\text{similarity}(A, B) = \frac{\sum_{i=1}^{n} A_i \times B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \times \sqrt{\sum_{i=1}^{n} B_i^2}}$$

#### Interpretação

| Valor | Significado |
|-------|-------------|
| **1.0** | Filmes idênticos (mesmo padrão de avaliações) |
| **0.0** | Filmes ortogonais (sem relação) |
| **-1.0** | Filmes opostos (padrões inversos) |

#### Implementação

```python
from sklearn.metrics.pairwise import cosine_similarity

# Transpõe a matriz para que filmes sejam linhas
movie_features = ratings_normalized.T.values  # shape: (22607, 5203)

# Calcula matriz de similaridade
similarity_matrix = cosine_similarity(movie_features)
# Resultado: matriz 22,607 × 22,607
```

#### Matriz de Similaridade Resultante

```
              filme_1  filme_2  filme_3  ...
filme_1         1.00     0.85     0.32  ...
filme_2         0.85     1.00     0.41  ...
filme_3         0.32     0.41     1.00  ...
...
```

---

## 6. Técnicas de Otimização

### 5.1 k-Nearest Neighbors (k-NN)

#### Problema

Usar **todos** os filmes similares para predição pode introduzir ruído de filmes pouco relacionados.

#### Solução

Limitar a predição aos **k filmes mais similares** que o usuário avaliou:

```python
K_NEIGHBORS = 30

# Ordena por similaridade (maior primeiro)
similarities.sort(reverse=True, key=lambda x: x[0])

# Pega apenas os k mais similares
top_k = similarities[:self.k_neighbors]
```

#### Benefício

- Reduz ruído de filmes fracamente relacionados
- Foca nos vizinhos mais relevantes
- Melhora a precisão das predições

### 5.2 Ajuste de Bias (User/Item Bias)

#### O Problema do Bias

Diferentes usuários e filmes têm tendências sistemáticas:

```
Crítico A: sempre dá notas altas (média pessoal: 4.5)
Crítico B: sempre dá notas baixas (média pessoal: 2.5)
Filme X: geralmente bem avaliado (média: 4.0)
Filme Y: geralmente mal avaliado (média: 2.0)
```

Sem ajuste, o modelo não captura essas tendências individuais.

#### A Solução: Modelo com Bias

A predição considera três componentes de bias:

$$\hat{r}_{ui} = \mu + b_u + b_i + \text{ajuste\_similaridade}$$

| Componente | Descrição | Exemplo |
|------------|-----------|---------|
| **μ (mu)** | Média global de todos os ratings | 3.28 |
| **b_u** | Bias do usuário (desvio da média global) | +0.5 (generoso) ou -0.3 (rigoroso) |
| **b_i** | Bias do item (desvio da média global) | +1.0 (filme popular) ou -0.5 (filme ruim) |

#### Cálculo dos Bias

```python
# 1. Média global
self.global_mean = all_ratings.mean()  # μ = 3.279

# 2. Bias do usuário: b_u = média_usuário - μ
user_means = ratings_matrix.mean(axis=1)
self.user_bias = user_means - self.global_mean

# 3. Bias do item: b_i = média_item - μ
item_means = ratings_matrix.mean(axis=0)
self.item_bias = item_means - self.global_mean
```

#### Valores Calculados

```
Média global (μ): 3.279
User bias range: [-1.587, +1.405]  (críticos muito rigorosos a muito generosos)
Item bias range: [-2.179, +1.435]  (filmes muito ruins a muito bons)
```

#### Normalização para Cálculo de Similaridade

Antes de calcular a similaridade, os ratings são **normalizados removendo os bias**:

```python
r_normalized = r_original - μ - b_u - b_i
```

Isso garante que a similaridade capture apenas a **relação intrínseca entre filmes**, não as tendências individuais de usuários ou a popularidade geral do filme.

### 5.3 Predição Final com Bias

A predição combina o **baseline** (bias) com o **ajuste de similaridade**:

```python
def predict_rating(self, user_ratings, movie_id, user_id):
    # 1. Calcula baseline (predição sem informação de similaridade)
    baseline = self.global_mean + self.item_bias[movie_id]
    if user_id in self.user_bias:
        baseline += self.user_bias[user_id]
    
    # 2. Para cada filme que o usuário avaliou, calcula o desvio
    similarities = []
    for rated_movie, rating in user_ratings.items():
        sim = self.item_similarity.loc[movie_id, rated_movie]
        if sim > 0:
            # Baseline do filme avaliado
            item_baseline = global_mean + item_bias[rated_movie] + user_bias[user_id]
            # Desvio = quanto o rating real diferiu do esperado
            deviation = rating - item_baseline
            similarities.append((sim, deviation))
    
    # 3. Usa apenas k vizinhos mais similares
    top_k = sorted(similarities, reverse=True)[:k_neighbors]
    
    # 4. Calcula ajuste ponderado pela similaridade
    adjustment = sum(sim * dev for sim, dev in top_k) / sum(sim for sim, _ in top_k)
    
    # 5. Predição final = baseline + ajuste
    prediction = baseline + adjustment
    
    # 6. Garante que está no intervalo [1, 5]
    return np.clip(prediction, 1, 5)
```

#### Intuição

1. **Baseline**: "Em média, esse usuário daria X para esse filme"
2. **Ajuste**: "Mas baseado em filmes similares que ele avaliou, ajustamos em ±Y"
3. **Predição**: Baseline + Ajuste

---

## 7. Avaliação do Modelo

### 6.1 Metodologia: Hold-Out Validation

Para cada usuário de teste:

1. **Separa** 20% dos ratings como conjunto de teste
2. **Usa** os 80% restantes para fazer predições
3. **Compara** predições com valores reais

```python
for user in sampled_users:
    user_ratings = ratings_matrix.loc[user].dropna()
    
    # Separa 20% para teste
    n_holdout = int(len(user_ratings) * 0.2)
    holdout_movies = np.random.choice(user_ratings.index, size=n_holdout)
    
    # Treina com os outros 80%
    train_ratings = {m: r for m, r in user_ratings.items() if m not in holdout_movies}
    
    # Prediz os 20% escondidos
    for movie in holdout_movies:
        predicted = recommender.predict_rating(train_ratings, movie, user_id=user)
        actual = user_ratings[movie]
        # Compara predicted vs actual
```

### 6.2 Métricas de Avaliação

#### RMSE (Root Mean Squared Error)

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(predicted_i - actual_i)^2}$$

- **Penaliza erros grandes** mais severamente (por causa do quadrado)
- Mesma unidade da escala de rating (1-5)
- **Quanto menor, melhor**

#### MAE (Mean Absolute Error)

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|predicted_i - actual_i|$$

- Erro médio absoluto
- Mais **robusto a outliers** que RMSE
- **Quanto menor, melhor**

#### Correlação de Pearson

$$r = \frac{\sum(x - \bar{x})(y - \bar{y})}{\sqrt{\sum(x-\bar{x})^2}\sqrt{\sum(y-\bar{y})^2}}$$

- Mede a **força da relação linear** entre predições e valores reais
- Varia de -1 a +1
- **Quanto mais próximo de 1, melhor**

---

## 8. Resultados

### 7.1 Evolução das Métricas

| Versão | RMSE | MAE | Correlação | Técnicas Aplicadas |
|--------|------|-----|------------|-------------------|
| Baseline | 0.9221 | 0.7501 | 0.3805 | CF básico |
| v2 | 0.8881 | 0.7095 | 0.4993 | + Filtros + k-NN (k=30) |
| **v3 (Final)** | **0.4849** | **0.3717** | **0.8819** | + User/Item Bias |

### 7.2 Melhoria Total

| Métrica | Valor Inicial | Valor Final | Melhoria |
|---------|---------------|-------------|----------|
| **RMSE** | 0.9221 | **0.4849** | **↓ 47.4%** |
| **MAE** | 0.7501 | **0.3717** | **↓ 50.4%** |
| **Correlação** | 0.3805 | **0.8819** | **↑ 131.8%** |

### 7.3 Interpretação dos Resultados

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **RMSE 0.48** | Excelente | Erro médio menor que **meio ponto** na escala 1-5 |
| **MAE 0.37** | Excelente | Em média, erramos por apenas **0.37 pontos** |
| **Correlação 0.88** | Muito Forte | Predições muito próximas dos valores reais |

### 7.4 Configuração Final do Sistema

```python
# Parâmetros de Filtragem
MIN_USER_RATINGS = 10   # Usuários com pelo menos 10 avaliações
MIN_MOVIE_RATINGS = 10  # Filmes com pelo menos 10 avaliações

# Parâmetros do Modelo
MIN_RATINGS = 5         # Mínimo de ratings para incluir filme na matriz
K_NEIGHBORS = 30        # Número de vizinhos para k-NN

# Técnicas Habilitadas
BIAS_ADJUSTMENT = True  # Ajuste de bias user/item
```

---

## 9. Fluxo Completo

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. CARREGAMENTO DOS DADOS                    │
├─────────────────────────────────────────────────────────────────┤
│  • Lê CSV do Rotten Tomatoes                                    │
│  • 1.44M reviews, 69K filmes, 15K críticos                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 2. PRÉ-PROCESSAMENTO                            │
├─────────────────────────────────────────────────────────────────┤
│  • Padroniza scores (frações, letras, %) → escala 1-5           │
│  • Remove duplicatas (mantém mais recente)                      │
│  • Filtra usuários/filmes com poucos ratings (≥10)              │
│  • Iterativo até convergência                                   │
│                                                                 │
│  Resultado: 1.2M reviews, 22K filmes, 5K usuários               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 3. MATRIZ DE RATINGS                            │
├─────────────────────────────────────────────────────────────────┤
│  • Cria matriz User × Item (5,203 × 22,607)                     │
│  • Células: ratings padronizados ou NaN                         │
│  • Esparsidade: 98.92%                                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 4. CÁLCULO DOS BIAS                             │
├─────────────────────────────────────────────────────────────────┤
│  • μ (média global): 3.279                                      │
│  • b_u (bias usuário): média_usuário - μ                        │
│  • b_i (bias item): média_item - μ                              │
│                                                                 │
│  • Normaliza ratings: r_norm = r - μ - b_u - b_i                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 5. MATRIZ DE SIMILARIDADE                       │
├─────────────────────────────────────────────────────────────────┤
│  • Calcula Cosine Similarity entre todos os pares de filmes     │
│  • Usa ratings normalizados (sem bias)                          │
│  • Resultado: matriz 22,607 × 22,607                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 6. PREDIÇÃO                                     │
├─────────────────────────────────────────────────────────────────┤
│  Para predizer rating do usuário U para filme F:                │
│                                                                 │
│  1. Baseline = μ + b_u + b_f                                    │
│  2. Encontra k=30 filmes mais similares a F que U avaliou       │
│  3. Para cada filme similar, calcula desvio do baseline         │
│  4. Ajuste = média ponderada dos desvios (peso = similaridade)  │
│  5. Predição = Baseline + Ajuste                                │
│  6. Clip para [1, 5]                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 7. RESULTADO FINAL                              │
├─────────────────────────────────────────────────────────────────┤
│  • RMSE: 0.4849 (erro < 0.5 pontos)                             │
│  • MAE: 0.3717                                                  │
│  • Correlação: 0.8819 (muito forte)                             │
│                                                                 │
│  ✅ Sistema pronto para recomendações!                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Referências

1. **Collaborative Filtering**: Resnick, P., et al. (1994). "GroupLens: An Open Architecture for Collaborative Filtering of Netnews"

2. **Item-Item CF**: Sarwar, B., et al. (2001). "Item-Based Collaborative Filtering Recommendation Algorithms"

3. **Bias in Recommender Systems**: Koren, Y. (2010). "Factor in the Neighbors: Scalable and Accurate Collaborative Filtering"

4. **Netflix Prize**: Bell, R., Koren, Y., Volinsky, C. (2007). "Modeling Relationships at Multiple Scales to Improve Accuracy of Large Recommender Systems"

---

*Documentação do Sistema de Recomendação - Novembro 2025*
