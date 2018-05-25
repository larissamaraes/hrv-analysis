"""

    Python module to extract statistics for Time series

    @author: Optimale, inc
    Update : 18-04-17
"""

import numpy as np
from sklearn.metrics import mean_squared_error
import itertools
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,StratifiedKFold
import random
from sklearn.metrics import roc_curve, auc,accuracy_score,precision_score,recall_score,confusion_matrix
from sklearn.pipeline import Pipeline


def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def fit_lstm(trainXlstm,trainYlstm,testXlstm,y_seq_data_lstm,scaler):
    model = Sequential()
    model.add(LSTM(10, input_dim=420))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainXlstm, trainYlstm, nb_epoch=10, batch_size=128, verbose=0)
    
    # fazendo o forecasting no conjunto de teste para definir o threshold dinamicamente
    # com relacao ao erro encontrado.
    print("definindo threshold ...")
    forecast_LSTM = general_forecasting(testXlstm, model,method='lstm', strategy_approach='sliding_windows', slidding_window_len=30)
    forecast_LSTM  = scaler.inverse_transform(forecast_LSTM)
    y_seq_data_lstm  = scaler.inverse_transform(y_seq_data_lstm)
    
    # definindo threshold
    threshold = mean_squared_error(y_seq_data_lstm,forecast_LSTM)
    
    return threshold, model


# funcao que retorna os forecasts para pontos com/sem ruido
def get_forecasts(model,data_matrix ,data_matrix_ruido, inicio_sem_ruidos, inicio_ruidos, observed_window, horizon):
    forecast_results=[]
    ind_values=[]
    forecasts=[]
    ruido_values, real_values = [],[]
    
    # analise sobre os ruidos
    for i in range(len(inicio_ruidos)):
        # boundary
        if (inicio_ruidos[i] - observed_window) < 0:
            continue
            
        values = data_matrix_ruido[inicio_ruidos[i] - observed_window: inicio_ruidos[i]]
        if(len(values) == 0):
            break
            
        value_reshaped = values.reshape(1,1,len(values))
        for val in xrange(0, horizon):
            result = model.predict([value_reshaped])
            value_reshaped = value_reshaped.tolist()
            newX = value_reshaped[0][0]
            newX.append(result[0][0])
            del value_reshaped[0][0][0]
            value_reshaped[0][0] = newX
            value_reshaped= np.array(value_reshaped)

            if inicio_ruidos[i]+val >= len(data_matrix_ruido):
                break
            forecast_results.append(result[0][0])
            ind_values.append(data_matrix_ruido[inicio_ruidos[i]+val])
        
        if (len(forecast_results) == horizon):
            forecasts.append(forecast_results)
            real_values.append(ind_values)
        forecast_results=[]
        ind_values = []
    
     # analise sobre os nao-ruidos
    for i in range(len(inicio_sem_ruidos)):   
        # boundary
        if (inicio_sem_ruidos[i] - observed_window) < 0:
            continue
        values = data_matrix[inicio_sem_ruidos[i] - observed_window: inicio_sem_ruidos[i]]
        if(len(values) == 0):
            break
            
        value_reshaped = values.reshape(1,1,len(values))
        for val in xrange(0, horizon):
            result = model.predict([value_reshaped])
            value_reshaped = value_reshaped.tolist()
            newX = value_reshaped[0][0]
            newX.append(result[0][0])
            del value_reshaped[0][0][0]
            value_reshaped[0][0] = newX
            value_reshaped= np.array(value_reshaped)

            if inicio_sem_ruidos[i]+val >= len(data_matrix):
                break
            forecast_results.append(result[0][0])
            ind_values.append(data_matrix[inicio_sem_ruidos[i]+val])
        
        if (len(forecast_results) == horizon):
            forecasts.append(forecast_results)
            real_values.append(ind_values)

        forecast_results=[]
        ind_values = []
    
    ruido_and_real = ruido_values + real_values
    
    
    return forecasts, ruido_and_real

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def calculate_metrics(errors, thresh):
    TP,TN,FP,FN=0.0,0.0,0.0,0.0
    for idx in range(len(errors)):
        # parte com ruidos 
        if idx < len(errors)/2:
            if errors[idx] > thresh:
                TP+=1.0
            else:
                FN+=1.0
        # parte sem ruidos
        else:
            if errors[idx] <= thresh:
                TN+=1.0
            else:
                FP+=1.0
    
    return TP,TN,FP,FN



def train_seqtoseq_lstm(trainXlstm,trainYlstm, pred_horizon):
    model = Sequential()
    model.add(LSTM(10, input_dim=420))
    model.add(Dense(pred_horizon))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainXlstm, trainYlstm, nb_epoch=100, batch_size=128, verbose=0)
    

    return model

# training svm and returning the classifier 

def train_model_noise_detection(dataframe):
    
    pontos_positivos = []
    pontos_negativos = []
    n_pontos = 200      
    
    train_size = int(len(dataframe) * 0.8)
    
    traininset = np.copy(dataframe[0:train_size])
    
    for idx in range(n_pontos):
        inicio = np.random.randint(0, train_size-60)
        if inicio + 60 < len(traininset):
            pontos_negativos.append(traininset[inicio: inicio + 60])

    d = np.std(traininset)                                        # Desvio padrao de todo o data_matrix                          

    inicio_ruido = np.random.randint(0, len(traininset)-60, n_pontos) # Gera n indices para iniciar o ruido

    duracao = np.random.normal(120, 40, n_pontos).round().astype(np.int)  # Gera n duracoes de ruidos

    std = 0.05 * d
    intensidade = np.abs(np.random.normal(d, std, n_pontos))                      # Gera n intensidades de ruidos

    for idx in range(n_pontos):
        if inicio_ruido[idx]-30 > 0 and inicio_ruido[idx]+30 < len(traininset):
            traininset[inicio_ruido[idx]: inicio_ruido[idx] + duracao[idx] + 1] += intensidade[idx] # Acrescenta a intensidade 
            pontos_positivos.append(traininset[inicio_ruido[idx] - 30: inicio_ruido[idx] + 30])  # Pontos positivos
    
    X_treino = pontos_negativos + pontos_positivos
    
    y = [-1] * len(pontos_negativos) + [1] * len(pontos_positivos)
         
    X_treino = np.array(X_treino)
    
    y = np.array(y).ravel()
    
    pipeline = Pipeline([
        ('preprocessing', preprocessing.StandardScaler()),
        ('classify', SVC(kernel='rbf',probability=True))
    ])

    params_list = {"classify__C": [0.01,1.0,10],'classify__gamma':  [0.1, 1, 10, 100] }
    #n_iter_search=20
    random_search = GridSearchCV(pipeline, param_grid=params_list, n_jobs=-1, cv=StratifiedKFold())
    random_search.fit(X_treino, y)
    clf_SVM = random_search
    print(random_search.best_params_)
    
    return clf_SVM

# evaluating the svm inducted 

def evaluate_model(clf, dataframe,dataframe_ruido,inicios_ruidos,inicios_sem_ruidos):
    pontos_negativos_teste = []
    pontos_positivos_teste = []
    train_size = int(len(dataframe) * 0.8)
    data_matrix_real = dataframe[train_size:]
    data_matrix_ruido = dataframe_ruido
    inicio_ruido = inicios_ruidos
    inicio_sem_ruido = inicios_sem_ruidos

    for idx in range(len(inicio_sem_ruido)):
        if inicio_ruido[idx]-30 > 0 and inicio_ruido[idx]+30 < len(data_matrix_real):
            pontos_negativos_teste.append(data_matrix_real[inicio_sem_ruido[idx]: inicio_sem_ruido[idx] + 60])
    
    for idx in range(len(inicio_ruido)):
        if inicio_ruido[idx]-30 > 0 and inicio_ruido[idx]+30 < len(data_matrix_ruido):
            pontos_positivos_teste.append(data_matrix_ruido[inicio_ruido[idx]-30: inicio_ruido[idx] + 30])
    
     
    X_teste = pontos_negativos_teste + pontos_positivos_teste
    
    y_teste = [-1] * len(pontos_negativos_teste) + [1] * len(pontos_positivos_teste)
         
    X_teste = np.array(X_teste)
    
    y_teste = np.array(y_teste).ravel()
    #preds2 = clf.predict(X_teste)
    #acc,prec,rec = accuracy_score(y_teste, preds), precision_score(y_teste, preds),recall_score(y_teste, preds)
    
    #conf = confusion_matrix(y_teste, preds)
    y_score_svm = clf.predict_proba(X_teste)[:,1]
    preds=[]
    
    # avaliando todos os thresholds possiveis com relacao
    # ao valor de accuracy
    
    
    
    
    # construindo a curva
    fpr_svm, tpr_svm, _ = roc_curve(y_teste, y_score_svm)
    roc_auc_svm = auc(fpr_svm, tpr_svm)
    
    accuracies=[]
    predictions=[]
    
    for threshold in _:
        individual_preds=[]
        for value in y_score_svm:
            if value > threshold:
                individual_preds.append(1)
            else:
                individual_preds.append(-1)
        accuracies.append(accuracy_score(y_teste, individual_preds))
        predictions.append(individual_preds)
    
    #preds = predictions[np.argmax(accuracies)]
    fpr = fpr_svm[np.argmax(accuracies)]
    tpr = tpr_svm[np.argmax(accuracies)]
    #conf = confusion_matrix(y_teste, preds)
    
    #tpr = float(conf[0][0])/float(np.sum(conf[0]))
    #fpr = float(conf[1][0])/float(np.sum(conf[1]))
    
    #print("Accuracy : ", acc)
    #print("Precision : ", prec)
    #print("Recall : ", rec)
    
    #cls = ["ruido","nao ruido"]
    #plt.figure()
    #plot_confusion_matrix(conf,cls,
    #                  title='Confusion matrix, without normalization')
    #plt.show()
    
    
    return fpr_svm, tpr_svm, roc_auc_svm,fpr,tpr, _


def noise_generator (data_matrix, two_way=False):
    n_pontos = 200  
    minutos = 60
    pontos_positivos = np.empty((n_pontos, minutos))
    pontos_negativos = np.empty((n_pontos, minutos))
    
    train_size = int(len(data_matrix) * 0.8)
    data_matrix = data_matrix[train_size:]
    
    minutos = data_matrix.shape[0]
    d = np.std(data_matrix)                                                             
    
    data_matrix_ruido = np.copy(data_matrix)
    
    inicio_ruido = np.random.randint(420, minutos, n_pontos) 
    inicio_sem_ruido = np.random.randint(420, minutos, n_pontos) 

    duracao = np.random.normal(120, 40, n_pontos).round().astype(np.int) 
    
    std = 0.05 * d
    intensidade = np.abs(np.random.normal(d, std, n_pontos))                      

    
    # gerando os pontos postivos
    if two_way:
        for i in range(n_pontos):
            if bool(random.getrandbits(1)):
                data_matrix_ruido[inicio_ruido[i]: inicio_ruido[i] + duracao[i] + 1] += intensidade[i] 
            else:
                data_matrix_ruido[inicio_ruido[i]: inicio_ruido[i] + duracao[i] + 1] -= intensidade[i]
    else:
        for i in range(n_pontos):
            data_matrix_ruido[inicio_ruido[i]: inicio_ruido[i] + duracao[i] + 1] += intensidade[i] 
                                                                                               
    return data_matrix_ruido, inicio_ruido, inicio_sem_ruido


def lstm_predictions(model,lastValue, testY, pred_horizon=6):
    predictions=[]
    forecast_results=[]
    yValues = []
    errors=[]
    
    X = lastValue
    for idx in xrange(0,len(X)-pred_horizon,pred_horizon):
        if idx + pred_horizon >= len(X):
            break

        auxX=X[idx].reshape(1,1,len(lastValue[0][0]))
        for val in xrange(idx, idx+pred_horizon):
                
            result = model.predict([auxX])
            auxX = auxX.tolist()
            newX = auxX[0][0]
            newX.append(result[0][0])
            del auxX[0][0][0]
            auxX[0][0] = newX
            auxX= np.array(auxX)

            forecast_results.append(result[0][0])
            
        test_value = testY[idx:idx+pred_horizon]
        predictions.append(forecast_results)
        yValues.append(test_value)
        
        forecast_results=[]
    
    return predictions, yValues


def general_forecasting(lastValue, classifier, size_forecasting=None,method=None, slidding_window_len=None, strategy_approach='retro',train=None):
    forecast = []
    forecast_window = []
    
    if strategy_approach == 'retro':
        if len(lastValue.shape) == 1:
            model = sm.tsa.statespace.SARIMAX(train, trend='n', order=(0,1,1),enforce_invertibility=False, seasonal_order=(0,1,1,48))
            mod = model.fit()
            forecastedArima = mod.get_forecast(steps=size_forecasting)
            forecastedArima = (forecastedArima.predicted_mean)
            newForecasted = []
            for i in forecastedArima:
                newForecasted.append(i)
            forecast = newForecasted
    
        elif len(lastValue.shape) == 2:
            X = lastValue[0]

            for item in range(size_forecasting) :
                result = classifier.predict(X)
                forecast.append(result[0])
                X = X.tolist()
                X.append(result[0])
                X = np.array(X[1:])
        
        elif len(lastValue.shape) == 3:
            X = lastValue[0].reshape(1,1,size_forecasting)

            for item in range(size_forecasting) :
                result = classifier.predict([X])
                forecast.append(result[0][0])
                X = X.tolist()
                newX = X[0][0]
                newX.append(result[0][0])
                del X[0][0][0]
                X[0][0] = newX
                X= np.array(X)
                            
    elif strategy_approach == 'sliding_windows':
        if len(lastValue.shape) == 2:    
            X = lastValue
            if slidding_window_len > 1:
                for idx in xrange(0,len(X)-slidding_window_len,slidding_window_len):
                    if idx + slidding_window_len > len(X):
                        break

                    auxX=X[idx]
                    for val in xrange(idx, idx+slidding_window_len):
                        result = classifier.predict([auxX])
                        auxX = auxX.tolist()
                        auxX.append(result[0])
                        auxX = np.array(auxX[1:])
                            
                        forecast.append(result[0])
                    #forecast.append(forecast_window)
                    #forecast_window=[]
            else:
                for item in X:
                    result = classifier.predict([item])
                    forecast.append(result[0])
        elif len(lastValue.shape) == 3:
            X = lastValue
            if slidding_window_len > 1:
                for idx in xrange(0,len(X)-slidding_window_len,slidding_window_len):
                    if idx + slidding_window_len > len(X):
                        break

                    auxX=X[idx].reshape(1,1,len(lastValue[0][0]))
                    for val in xrange(idx, idx+slidding_window_len):
                        #value_reshaped = X[val].reshape(1,1,len(lastValue[0][0]))
                        #result = classifier.predict([value_reshaped])
                        result = classifier.predict([auxX])
                        auxX = auxX.tolist()
                        newX = auxX[0][0]
                        newX.append(result[0][0])
                        del auxX[0][0][0]
                        auxX[0][0] = newX
                        auxX= np.array(auxX)
                        
                        forecast.append(result[0][0])
                    #forecast.append(forecast_window)
                    #forecast_window=[]
            else:
                for item in X:
                    val = item.reshape(1,1,len(lastValue[0][0]))
                    result = classifier.predict([val])
                    forecast.append(result[0][0])
    
    
    return np.array(forecast)
    

# smoothing an array that represents the time serie.
def smooth(x, window_len, window='hanning'):


    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]



def readData(inputStringCSV):
    dataframe = pandas.read_csv(inputStringCSV, engine='python', skipfooter=3)
    return dataframe

def generate_datasets(dataset,method='slidding_windows',look_back=1,pred_horizon=None):
    dataX, dataY = [], []
    train_data, test_data = [], []
    train, test=[],[]
    seq_y, dataYSeq=[],[]
    
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    
    for i in range(len(dataset[0:train_size])-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    train_data = [np.array(dataX), np.array(dataY)]
    
    dataX, dataY = [],[]
    
    if method == 'retro':
        for i in xrange(len(dataset[:train_size]),len(dataset)-look_back):
            a = dataset[i:(i+look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        test_data = [np.array(dataX), np.array(dataY)]
    else:
        for i in xrange(len(dataset[:train_size-look_back-2]),len(dataset)-look_back):
            a = dataset[i:(i+look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])

        test_data = [np.array(dataX), np.array(dataY)]
        if not pred_horizon == None:
            for idx in range(0, len(dataY)-pred_horizon, pred_horizon):
                if idx + pred_horizon > len(dataY):
                    break

                for val in xrange(idx, idx+pred_horizon):
                    dataYSeq.append(dataY[val])

        return train_data,test_data, np.array(dataYSeq)
    
    return train_data,test_data


