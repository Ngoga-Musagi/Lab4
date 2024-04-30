import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import io
from lab4_header import scores_table
from post_score import post_score
import json

print('starting up iris model service')

global models, datasets,metrics
models = []
datasets = []
metrics = []

def build():
    global models

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
        ])

    model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    models.append( model )
    model_ID = len(models) - 1

    return model_ID

def load_local():
    global datasets

    print("load local default data")

    dataFolder = './iris_extended_encoded.csv'
    dataFile = dataFolder + "iris.data"

    datasets.append( pd.read_csv(dataFile) )
    return len( datasets ) - 1

def add_dataset( df ):
    global datasets

    datasets.append( df )
    return len( datasets ) - 1

def get_dataset( dataset_ID ):
    global datasets

    return datasets[dataset_ID]

def train(model_ID, dataset_ID):
    global datasets, models
    dataset = datasets[dataset_ID]
    model = models[model_ID]

    X = dataset.iloc[:, 1:].values

    y = dataset.iloc[:,0].values

    encoder =  LabelEncoder()
    y1 = encoder.fit_transform(y)
    Y = pd.get_dummies(y1).values

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    history = model.fit(X_train, y_train, batch_size=1, epochs=10)
    print(history.history)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    y_pred = model.predict(X_test)

    actual = np.argmax(y_test,axis=1)
    predicted = np.argmax(y_pred,axis=1)
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")

    conf_matrix = confusion_matrix(actual, predicted)
    print('Confusion matrix on test data is {}'.format(conf_matrix))
    print('Precision Score on test data is {}'.format(precision_score(actual, predicted, average=None)))
    print('Recall Score on test data is {}'.format(recall_score(actual, predicted, average=None)))

    return(history.history)

def score( model_ID, elevation=0.18, soil_type=0.15, sepal_length=0.125, sepal_width=0.1458,
       petal_length=0.77, petal_width=0.96, sepal_area=0.789, petal_area=0.15,
       sepal_aspect_ratio=0.1741, petal_aspect_ratio=0.34,
       sepal_to_petal_length_ratio=0.412, sepal_to_petal_width_ratio=0.1785,
       sepal_petal_length_diff=0.966, sepal_petal_width_diff=0.1123,
       petal_curvature_mm=0.14124, petal_texture_trichomes_per_mm2=0.178,
       leaf_area_cm2=0.1102, sepal_area_sqrt=0.1012, petal_area_sqrt=0.17778,area_ratios=0.6):
    global models
    model = models[model_ID]

    # x_test2 = [ [elevation,soil_type,sepal_length,sepal_width,petal_length,petal_width,sepal_area,petal_area,sepal_aspect_ratio,petal_aspect_ratio,sepal_to_petal_length_ratio,sepal_to_petal_width_ratio,sepal_petal_length_diff,sepal_petal_width_diff,petal_curvature_mm,petal_texture_trichomes_per_mm2,leaf_area_cm2,sepal_area_sqrt,petal_area_sqrt,area_ratios] ]
    x_test2 = [ [elevation,soil_type,sepal_length,sepal_width,petal_length,petal_width,sepal_area,petal_area,sepal_aspect_ratio,petal_aspect_ratio,sepal_to_petal_length_ratio,sepal_to_petal_width_ratio,sepal_petal_length_diff,sepal_petal_width_diff,petal_curvature_mm,petal_texture_trichomes_per_mm2,leaf_area_cm2,sepal_area_sqrt,petal_area_sqrt,area_ratios] ]
    x_test2 = np.array([[0.2, 0.3, 0.19, 0.1458, 0.97, 0.56, 0.309, 0.75, 0.941, 0.64, 0.412, 0.1785, 0.76, 0.2123, 0.44124, 0.178, 0.32, 0.1012,0.3778,0.9]])
    y_pred2 = model.predict(x_test2)
    print(y_pred2)
    iris_class = np.argmax(y_pred2, axis=1)[0]
    print(iris_class)

    return str(iris_class)
# def test(model_ID, dataset_ID):
#     global datasets, models, metrics
#     dataset = datasets[dataset_ID]
#     model = models[model_ID]

#     X = dataset.iloc[:, 1:].values
#     y = dataset.iloc[:, 0].values

#     encoder = LabelEncoder()
#     y1 = encoder.fit_transform(y)
#     Y = pd.get_dummies(y1).values

#     X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

#     loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
#     print('Test loss:', loss)
#     print('Test accuracy:', accuracy)

#     y_pred = model.predict(X_test)

#     actual = np.argmax(y_test, axis=1)
#     predicted = np.argmax(y_pred, axis=1)
#     print(f"Actual: {actual}")
#     print(f"Predicted: {predicted}")

#     conf_matrix = confusion_matrix(actual, predicted)
#     print('Confusion matrix on test data is {}'.format(conf_matrix))
#     print('Precision Score on test data is {}'.format(precision_score(actual, predicted, average=None)))
#     print('Recall Score on test data is {}'.format(recall_score(actual, predicted, average=None)))

#     # Saving test results to metrics list
#     metrics.append({
#         'model_ID': model_ID,
#         'dataset_ID': dataset_ID,
#         'test_loss': loss,
#         'test_accuracy': accuracy,
#         'confusion_matrix': conf_matrix,
#         'precision_score': precision_score(actual, predicted, average=None),
#         'recall_score': recall_score(actual, predicted, average=None)
#     })

#     return {
#         'test_loss': loss,
#         'test_accuracy': accuracy,
#         'confusion_matrix': conf_matrix.tolist(),
#         'precision_score': precision_score(actual, predicted, average=None).tolist(),
#         'recall_score': recall_score(actual, predicted, average=None).tolist()
#     }

def test(model_ID, dataset_ID):
    """ Tests the specified model using the specified dataset. """
    global datasets, models
    dataset = datasets[dataset_ID]
    model = models[model_ID]

    X = dataset.iloc[:, 1:].values  
    y = dataset.iloc[:, 0].values   

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    Y = pd.get_dummies(y_encoded).values

    _, X_test, _, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    y_pred = model.predict(X_test)

    results = []
    for i in range(len(X_test)):
        
        features_dict = dict(zip(dataset.columns[1:], X_test[i]))

        species_label = encoder.inverse_transform([np.argmax(y_test[i])])[0]
        features_dict = update_feature_dict(features_dict, 'species', species_label)


        # Ensure correct formatting of dictionary keys as string literals
        feature_string = json.dumps(features_dict)  # Convert dictionary to JSON string ensuring order
        class_string = str(np.argmax(y_pred[i]))
        actual_string = str(encoder.inverse_transform([np.argmax(y_test[i])])[0])  # Get original label from encoded
        prob_string = str(max(y_pred[i]))

       
        response = post_score(scores_table, feature_string, class_string, actual_string, prob_string)

        results.append({
            'features': feature_string,
            'class': class_string,
            'actual_class': actual_string,
            'probability': prob_string,
            'dynamodb_response': response
        })
    return results
def update_feature_dict(orginal_dict, key, value):
# Create a new dictionary with the new key-value pair
    dict1 = {key: value}
    dict1.update(orginal_dict)
    
    return dict1
