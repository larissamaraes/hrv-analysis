from flask import Flask, jsonify, json, request
from class_ruidos import classificador_ruidos

app = Flask(__name__)

@app.route('/classruidos', methods = ['POST'])
def classruidos():

    data_teste = request.get_json()

    bool, score_treino = classificador_ruidos(data_teste)

    if (bool == 1):
        return json.dumps({'Score_treino': score_treino, 'Ruido': 'Yes'})
    else:
        return json.dumps({'Score_treino': score_treino , 'Ruido': 'No'})

if __name__ == '__main__':
    app.run(debug=True)