# Importando as bibliotecas necessárias
%matplotlib inline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
import numpy as np
import csv

# Lendo os dados da planilha e carregando em um objeto
dados = pd.read_table('Máscara de dados.csv',sep=',',delimiter=';')

# Verificando o formato do dataset
print('Nº de Linhas:',format(np.shape(dados)[0]))
print('Nº de Colunas:',format(np.shape(dados)[1]))

# Visualizando as primeiras vinte observações
dados.head(20)

teste = [np.array(dados[['POPE','PFIN','PLOG','PMKT','PRH','PSI','SEXO','IDADE']]),dados['ESTILO_CAT']]

teste[0].shape

# Verificando a divisão do dataset
print('Treino: {}'.format(X_train.shape))
print('Teste: {}'.format(X_test.shape))

# Elaborando o modelo de classificação com base em KNN
knn = KNeighborsClassifier(n_neighbors=3, algorithm='brute',weights='distance')
knn.fit(X_train,y_train)

# Avaliando o modelo criado
print('Treino: {}'.format(knn.score(X_train,y_train)))
print('Teste: {}'.format(knn.score(X_test,y_test)))

# Elaborando o modelo de classificação com base em Regressão Logística
logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train,y_train)


# Avaliando o modelo criado
print('Treino: {}'.format(logreg.score(X_train,y_train)))
print('Teste: {}'.format(logreg.score(X_test,y_test)))

# Elaborando o modelo de classificação com base em Árvore de Decisão 
dectree = DecisionTreeClassifier()
dectree.fit(X_train,y_train)

# Avaliando o modelo criado
print('Treino: {}'.format(dectree.score(X_train,y_train)))
print('Teste: {}'.format(dectree.score(X_test,y_test)))

# Elaborando o modelo de classificação com base em Random Forest
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

# Avaliando o modelo criado
print('Treino: {}'.format(rf.score(X_train,y_train)))
print('Teste: {}'.format(rf.score(X_test,y_test)))

# Elaborando o modelo de classificação com base em Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train,y_train)

# Avaliando o modelo criado
print('Treino: {}'.format(gnb.score(X_train,y_train)))
print('Teste: {}'.format(gnb.score(X_test,y_test)))
