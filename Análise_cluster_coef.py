import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Carregar os dados
df = pd.read_excel(r'C:\Users\a475039\Desktop\TCC Dataset\PR_2019_2023.xlsx')

# Criar a coluna 'gravidade' com base nas condições especificadas
condicoes = [
    (df['mortos'] > 0) | (df['feridos_graves'] > 0),  # Condição para "grave"
    (df['feridos_leves'] > 0) & (df['mortos'] == 0) & (df['feridos_graves'] == 0),  # Condição para "moderado"
    (df['feridos_leves'] == 0) & (df['feridos_graves'] == 0) & (df['mortos'] == 0)  # Condição para "leve"
]
valores = ['grave', 'moderado', 'leve']
df['gravidade'] = np.select(condicoes, valores, default='indefinido')

# Remover casos indefinidos, se houver
df = df[df['gravidade'] != 'indefinido']

# Tratar valores ausentes nas colunas relevantes
df = df.dropna(subset=['feridos_leves', 'feridos_graves', 'mortos', 'longitude', 'latitude', 'causa_acidente'])

# Selecionar as colunas para clustering (latitude e longitude)
data = df[['latitude', 'longitude']]

# Padronizar os dados antes de aplicar o KMeans (recomendado para coordenadas geográficas)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Aplicação do KMeans
optimal_k = 4  # número ideal de clusters
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

# Agora, para cada cluster, realizaremos a análise de regressão logística
for cluster_name in cluster_names.values():
    print(f"\nAnálise para o Cluster {cluster_name}:")

    # Filtrar os dados para o cluster atual
    df_cluster = df[df['cluster'] == cluster_name]

    # Selecionar apenas as colunas 'causa_acidente' e 'gravidade'
    df_cluster = df_cluster[['causa_acidente', 'gravidade']]

    # Remover categorias raras em 'causa_acidente'
    contagem_causas = df_cluster['causa_acidente'].value_counts()
    causas_frequentes = contagem_causas[contagem_causas > 10].index  # Ajuste o limite conforme necessário
    df_cluster = df_cluster[df_cluster['causa_acidente'].isin(causas_frequentes)]

    # Verificar se há dados suficientes para análise
    if df_cluster.shape[0] < 50:  # Ajuste este valor conforme necessário
        print(f"Dados insuficientes no Cluster {cluster_name} para análise confiável.")
        continue

    # Codificar 'causa_acidente' usando One Hot Encoding
    df_encoded = pd.get_dummies(df_cluster, columns=['causa_acidente'], drop_first=True)

    # Codificar a variável alvo 'gravidade' usando LabelEncoder
    le = LabelEncoder()
    df_encoded['gravidade'] = le.fit_transform(df_encoded['gravidade'])

    # Dividir os dados em variáveis independentes (X) e variável dependente (y)
    X = df_encoded.drop('gravidade', axis=1)
    y = df_encoded['gravidade']

    # Verificar se há pelo menos duas classes na variável alvo
    if len(np.unique(y)) < 2:
        print(f"Variável alvo tem apenas uma classe no Cluster {cluster_name}. Não é possível realizar a regressão logística.")
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200, class_weight='balanced')

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    if 'grave' in le.classes_:
        class_index = np.where(le.classes_ == 'grave')[0][0]
        grave_coefficients = model.coef_[class_index]
        # Obter os 10 maiores coeficientes que mais afetam os acidentes graves
        coef_series = pd.Series(grave_coefficients, index=X.columns)
        top_10_coefficients = coef_series.nlargest(10)
        print("\nTop 10 maiores coeficientes que mais afetam os acidentes graves:")
        print(top_10_coefficients)
    else:
        print(f"A classe 'grave' não está presente no Cluster {cluster_name}.")
