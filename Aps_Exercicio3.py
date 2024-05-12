# Importação de bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Carregamento dos dados
df = pd.read_csv('Banco1.csv')

# Análise exploratória de dados (EDA)
def eda(df):
    # Estatísticas descritivas
    print("Estatísticas descritivas:")
    print(df.describe().T.round(2))
    
    # Correlações
    print("\nCorrelações:")
    print(df.corr(numeric_only=True))
    
    # Distribuição de variáveis categóricas
    for col in df.select_dtypes(include=['object']).columns:
        print(f"\nValores únicos em {col}: {df[col].unique()}")
    
    # Visualização da distribuição da variável alvo
    sns.countplot(x='aderiu_emprestimo', data=df)
    plt.title('Distribuição da adesão ao empréstimo')
    plt.show()

eda(df)

# Pré-processamento dos dados
def preprocessamento(df):
    # Separar variáveis explicativas (X) e variável alvo (y)
    X = df.drop('aderiu_emprestimo', axis=1)
    y = df['aderiu_emprestimo']
    
    # Codificação da variável alvo
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Separar colunas numéricas e categóricas
    numeric_features = X.select_dtypes(include=[np.number]).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Pipeline para colunas numéricas
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    # Pipeline para colunas categóricas
    categorical_transformer = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore'))])
    
    # Column Transformer para aplicar pipelines nas colunas corretas
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    
    # Aplicar pré-processamento aos dados
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, preprocessor

X, y, preprocessor = preprocessamento(df)

# Divisão dos dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Treinamento e avaliação dos modelos
def treinar_avaliar(X_train, y_train, X_test, y_test):
    # Dicionário com os classificadores a serem avaliados
    classificadores = {
        'Random Forest': RandomForestClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Regressão Logística': LogisticRegression(random_state=42)
    }
    
    resultados = []

    # Treinar e avaliar cada modelo
    for nome, classificador in classificadores.items():
        classificador.fit(X_train, y_train)
        y_pred = classificador.predict(X_test)
        acuracia = accuracy_score(y_test, y_pred)
        f1_score = classification_report(y_test, y_pred, output_dict=True)
        
        resultados.append({
            'Modelo': nome,
            'Acurácia': acuracia,
            'F1-Score': f1_score['weighted avg']['f1-score']
        })
    
    return pd.DataFrame(resultados)

resultados_df = treinar_avaliar(X_train, y_train, X_test, y_test)
print(resultados_df)

# Melhor modelo baseado em acurácia
melhor_modelo = resultados_df[resultados_df['Acurácia'] == resultados_df['Acurácia'].max()]['Modelo'].values[0]

print(f"\nO melhor modelo é: {melhor_modelo}")

# Avaliação final do melhor modelo
def avaliar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de confusão')
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.show()
    
    # Relatório de classificação
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))

# Selecionar o melhor modelo
if melhor_modelo == 'Random Forest':
    melhor_modelo = RandomForestClassifier(random_state=42)
elif melhor_modelo == 'K-Nearest Neighbors':
    melhor_modelo = KNeighborsClassifier()
elif melhor_modelo == 'Regressão Logística':
    melhor_modelo = LogisticRegression(random_state=42)

# Treinar o melhor modelo e avaliar
melhor_modelo.fit(X_train, y_train)
avaliar_modelo(melhor_modelo, X_test, y_test)
