def classificador_ruidos(data_teste):

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
    from sklearn.svm import SVC
    from flask import json

    # ________________________________________________________________________

    # Preparando os dados para treino

    data = pd.read_csv('690a-vazao-vila-aurea-20-04-2016.csv')

    data_matrix = data.as_matrix()

    data_matrix = data_matrix[340519:, 2:]

    minutos = data_matrix.shape[0]

    data_matrix = np.array(data_matrix).ravel()

    # _________________________________________________________________________

    # Eliminando os buracos

    for i in range(minutos):
        if (pd.isnull(data_matrix[i])):
            j = i
            while (i < minutos and pd.isnull(data_matrix[i])):
                i += 1

            data_matrix[j:i] = (data_matrix[j - 1] + data_matrix[i]) / 2

    # ________________________________________________________________________

    # Funcao de suavizacao da serie

    def hanning(x, window_len):
        # x_=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
        # Extend input data to avoid border issues.
        x_ = np.r_[[x[0]] * (window_len - 1), x, [x[-1]] * (window_len - 1)]
        # Hanning window (filter).
        f = np.hanning(window_len)
        # Apply convolution.
        y = np.convolve(x_, f / f.sum(), mode='same')
        # Return data to original size.

        return y[window_len - 1:-window_len + 1]

    # ________________________________________________________________________

    # Dados da vazao suavizados

    data_matrix_ruido = hanning(data_matrix, 10)

    # ________________________________________________________________________

    # Gerar 100 pontos negativos (sem ruido) para treino

    n_pontos_negativos = 100
    minutos_sem_ruido = 60

    pontos_negativos = np.empty((n_pontos_negativos, minutos_sem_ruido))

    for i in range(n_pontos_negativos):
        inicio = np.random.randint(0, minutos - minutos_sem_ruido)
        pontos_negativos[i,] = data_matrix_ruido[inicio: inicio + minutos_sem_ruido]

    pontos_negativos = pontos_negativos.reshape(1, -1)

    # ________________________________________________________________________

    # Gerar 100 pontos positivos (com ruido) para treino


    n_pontos_positivos = 100
    minutos_ruido = 60
    pontos_positivos = np.empty((n_pontos_positivos, minutos_ruido))

    d = np.std(data_matrix)  # Desvio padrao do data_matrix

    inicio_ruido = np.random.randint(0, minutos, n_pontos_positivos)  # Gera n indices para iniciar o ruido

    duracao = np.random.normal(120, 40, n_pontos_positivos).round().astype(np.int)  # Gera n duracoes de ruidos

    std = 0.05 * d
    intensidade = np.abs(np.random.normal(d, std, n_pontos_positivos))  # Gera n intensidades de ruidos

    for i in range(n_pontos_positivos):
        data_matrix_ruido[inicio_ruido[i]: inicio_ruido[i] + duracao[i] + 1] += intensidade[
            i]  # Acrescenta a intensidade
        # gerada para cada ruido

    for i in range(n_pontos_positivos):
        pontos_positivos[i,] = data_matrix[inicio_ruido[i] - 30: inicio_ruido[i] + 30]  # Pontos positivos

    pontos_positivos = pontos_positivos.reshape(1, -1)

    # ________________________________________________________________________

    # Treino com o data_matrix da vazao da Vila Aurea

    X_treino = [pontos_negativos, pontos_positivos]
    y_treino = [-1] * pontos_negativos.shape[1] + [1] * pontos_positivos.shape[1]

    X_treino = np.array(X_treino).ravel()
    y_treino = np.array(y_treino).ravel()

    X_treino = X_treino.reshape(-1,1)

    params = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10, 100]}

    random_search = RandomizedSearchCV(SVC(kernel = 'rbf'), param_distributions = params, cv = StratifiedKFold())

    random_search.fit(X_treino, y_treino)
    
    # clf = random_search.best_estimator_
    # joblib.dump(clf, "meu_classificaro.pkl")
    
    # from sklearn.externals import joblib
    # clf = load_model("meu_classificador.pkl")

    # ________________________________________________________________________

    # Teste com o data_teste

    data_teste = json.loads(data_teste)

    data_teste = np.array(data_teste).ravel()

    data_teste = data_teste.reshape(-1,1)

    y_predict = random_search.predict(data_teste)

    score_treino = random_search.best_score_

    #score_teste = random.search.score(data_teste)

    if (max(y_predict) == 1):
        return 1, score_treino          # Tem ruido no data_teste
    else:
        return -1, score_treino         # Nao tem ruido no data_teste