# iris_api.py
import pandas as pd
import io
from flask import Flask, request, jsonify
from base_iris_lab1 import build, load_local, add_dataset, train, score,test
from post_score import post_score
from lab4_header import scores_table

app = Flask(__name__)


@app.route('/iris/datasets', methods=['POST'])
def upload_dataset():
    
    train_data = request.files['train']
    dataset = pd.read_csv(train_data)
    dataset_index = add_dataset(dataset)

    return jsonify({'index': dataset_index})

@app.route('/iris/model', methods=['POST'])
def build_and_train():
    
    dataset_index = int(request.form['dataset'])

    model_index = build()
    history = train(model_index, dataset_index)

    return jsonify({'index': model_index})

@app.route('/iris/model/<int:model_index>', methods=['PUT'])
def retrain_model(model_index):
    
    dataset_index = int(request.form['dataset'])
    
    if dataset_index is None:
        return jsonify({'ERROR': 'Dataset index not provided'}), 400

    try:
        dataset_index = int(dataset_index)
    except ValueError:
        return jsonify({'ERROR': 'Invalid dataset index'}), 400
    # Re-train the model using the dataset
    history = train(model_index, dataset_index)
    # Return the learning  history
    return jsonify({'learning_history': history})


@app.route('/iris/model/<int:model_index>', methods=['GET'])
def score_model(model_index):

    values_string = request.args.get('values')  # Assuming the values are passed under the key 'valuess'
    features = [float(value) for value in values_string.split(',')]
    score_result = score(model_index, *features)
    
    return jsonify({'result': score_result})

@app.route('/iris/model/<int:model_index>/test', methods=['GET'])
def test_model(model_index):
    dataset_index = request.args.get('dataset')

    if dataset_index is None:
        return jsonify({'error': 'Dataset index not provided'}), 400

    try:
        dataset_index = int(dataset_index)
    except ValueError:
        return jsonify({'error': 'Invalid dataset index'}), 400

    test_results = test(model_index, dataset_index)
    return jsonify(test_results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=4000)
    
