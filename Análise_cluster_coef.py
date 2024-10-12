import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Função para calcular Odds Ratio
def calculate_odds_ratio(coefficients):
    return np.exp(coefficients)

# Carregar os dados
df = pd.read_excel(r'C:\Users\a475039\Desktop\TCC Dataset\PR_2019_2023.xlsx')

# Criar a coluna 'gravidade' com base nas novas condições especificadas
condicoes = [
    (df['mortos'] > 0) | (df['feridos_graves'] > 0) | (df['feridos_leves'] > 0),  # Condição para "grave"
    (df['feridos_leves'] == 0) & (df['feridos_graves'] == 0) & (df['mortos'] == 0)  # Condição para "leve"
]
valores = ['grave', 'leve']
df['gravidade'] = np.select(condicoes, valores, default='indefinido')

# Remover casos indefinidos
df = df[df['gravidade'] != 'indefinido']

# Tratar valores ausentes nas colunas relevantes
df = df.dropna(subset=['feridos_leves', 'feridos_graves', 'mortos', 'longitude', 'latitude', 'causa_acidente'])

# Selecionar as colunas para clustering (latitude e longitude)
data = df[['latitude', 'longitude']]

# Padronizar os dados antes de aplicar o KMeans
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Aplicação do KMeans
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# Substituir os números dos clusters pelos nomes fornecidos
cluster_names = {0: 'Região Norte', 1: 'Curitiba e Litoral', 2: 'Região Oeste', 3: 'Campos Gerais'}
df['cluster'] = [cluster_names[cluster] for cluster in clusters]

# Visualização geográfica dos clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='longitude', y='latitude', hue='cluster', data=df, palette='Set1')
plt.title('Clusters Geográficos de Acidentes')
plt.show()

# Dicionário para armazenar a precisão de cada cluster
cluster_precisions = []

# Agora, para cada cluster, realizaremos a análise de regressão logística binária
for cluster_name in cluster_names.values():
    print(f"\nAnálise para o Cluster {cluster_name}:")

    # Filtrar os dados para o cluster atual
    df_cluster = df[df['cluster'] == cluster_name]

    # Selecionar apenas as colunas 'causa_acidente' e 'gravidade'
    df_cluster = df_cluster[['causa_acidente', 'gravidade']]

    # Remover categorias raras em 'causa_acidente'
    contagem_causas = df_cluster['causa_acidente'].value_counts()
    causas_frequentes = contagem_causas[contagem_causas > 10].index
    df_cluster = df_cluster[df_cluster['causa_acidente'].isin(causas_frequentes)]

    # Verificar se há dados suficientes para análise
    if df_cluster.shape[0] < 50:
        print(f"Dados insuficientes no Cluster {cluster_name} para análise confiável.")
        continue

    # Codificar 'causa_acidente' usando One Hot Encoding
    df_encoded = pd.get_dummies(df_cluster, columns=['causa_acidente'], drop_first=True)

    # Codificar a variável alvo 'gravidade' usando LabelEncoder (grave = 1, leve = 0)
    le = LabelEncoder()
    df_encoded['gravidade'] = le.fit_transform(df_encoded['gravidade'])

    # Dividir os dados em variáveis independentes (X) e variável dependente (y)
    X = df_encoded.drop('gravidade', axis=1)
    y = df_encoded['gravidade']

    # Verificar se há pelo menos duas classes na variável alvo
    if len(np.unique(y)) < 2:
        print(f"Variável alvo tem apenas uma classe no Cluster {cluster_name}. Não é possível realizar a regressão logística.")
        continue

    # Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Modelo de regressão logística binária
    model = LogisticRegression(solver='lbfgs', max_iter=200, class_weight='balanced')

    model.fit(X_train, y_train)

    # Previsões
    y_pred = model.predict(X_test)

    # Relatório de classificação
    print("\nRelatório de Classificação:")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Salvar a precisão das classes "grave" e "leve"
    precision_grave = report['grave']['precision']
    precision_leve = report['leve']['precision']
    cluster_precisions.append({'Cluster': cluster_name, 'Precisão Grave': precision_grave, 'Precisão Leve': precision_leve})

    # Gerar a matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    # Adicionar as precisões de ambas as classes à matriz de confusão
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.loc['Precisão'] = [f'{precision_grave:.4f}', f'{precision_leve:.4f}']  # Adiciona as precisões de "grave" e "leve"

    # Salvar a matriz de confusão com as precisões em Excel
    cm_df.to_excel(f'matriz_confusao_{cluster_name}.xlsx')

    # Coeficientes da classe 'grave'
    if 'grave' in le.classes_:
        grave_coefficients = model.coef_[0]
        odds_ratios = calculate_odds_ratio(grave_coefficients)
        # Criar DataFrame com variáveis, coeficientes e odds ratios
        coef_df = pd.DataFrame({
            'Variável': X.columns,
            'Coeficiente': grave_coefficients,
            'Odds Ratio': odds_ratios
        })
        # Ordenar pelos top 10 maiores coeficientes
        top_10_coef_df = coef_df.nlargest(10, 'Coeficiente')

        # Salvar os coeficientes e odds ratios em Excel
        top_10_coef_df.to_excel(f'coeficientes_odds_{cluster_name}.xlsx', index=False)

        print("\nTop 10 maiores coeficientes que mais afetam os acidentes graves:")
        print(top_10_coef_df)
    else:
        print(f"A classe 'grave' não está presente no Cluster {cluster_name}.")

# Criar DataFrame com as precisões de cada cluster
precision_df = pd.DataFrame(cluster_precisions)

# Salvar a tabela resumo de precisão em Excel
precision_df.to_excel('precisao_clusters.xlsx', index=False)

print("\nTabela resumo de precisão de cada cluster:")
print(precision_df)
