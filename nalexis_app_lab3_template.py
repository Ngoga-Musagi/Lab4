# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
from dash import Dash, dcc, html, Input, Output, State
from dash import Dash, dash_table

tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

col_style = {'display':'grid', 'grid-auto-flow': 'row'}
row_style = {'display':'grid', 'grid-auto-flow': 'column'}

import plotly.express as px
import pandas as pd

import requests

app = Dash(__name__)
global df
df = pd.read_csv("iris_extended_encoded.csv",sep=',')
df_csv = df.to_csv(index=False)


BASE_URL = "http://localhost:4000/iris"
app.layout = html.Div(children=[
    html.H1(children='Iris classifier'),
    dcc.Tabs([
    dcc.Tab(label="Explore Iris training data", style=tab_style, selected_style=tab_selected_style, children=[

    html.Div([
        html.Div([
            html.Label(['File name to Load for training or testing'], style={'font-weight': 'bold'}),
            dcc.Input(id='file-for-train', type='text', style={'width':'100px'}),
            html.Div([
                html.Button('Load', id='load-val', style={"width":"60px", "height":"30px"}),
                html.Div(id='load-response', children='Click to load')
            ], style=col_style)
        ], style=col_style),

        html.Div([
            html.Button('Upload', id='upload-val', style={"width":"60px", "height":"30px"}),
            html.Div(id='upload-response', children='Click to upload')
        ], style=col_style| {'margin-top':'20px'})

    ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),


html.Div([
    html.Div([
        html.Div([
            html.Label(['Feature'], style={'font-weight': 'bold'}),
            dcc.Dropdown(
                df.columns[1:], #<dropdown values for histogram>
                df.columns[1],           #<default value for dropdown>
                id='hist-column'
            )
            ], style=col_style ),
        dcc.Graph( id='selected_hist' )
    ], style=col_style | {'height':'400px', 'width':'400px'}),

    html.Div([

    html.Div([

    html.Div([
        html.Label(['X-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
            df.columns[1:], #<dropdown values for scatter plot x-axis>
            df.columns[1],           #<default value for dropdown>
            id='xaxis-column'
            )
        ]),

    html.Div([
        html.Label(['Y-Axis'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
               df.columns[1:], #<dropdown values for scatter plot y-axis>
               df.columns[1],           #<default value for dropdown>
            id='yaxis-column'
            )
        ])
    ], style=row_style | {'margin-left':'50px', 'margin-right': '50px'}),

    dcc.Graph(id='indicator-graphic')
    ], style=col_style)
], style=row_style),

    html.Div(id='tablecontainer', children=[
        dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], page_size=15,
            id='datatable' )
        ])
    ]),
    dcc.Tab(label="Build model and perform training", id="train-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Button('New model', id='build-val', style={'width':'90px', "height":"30px"}),
                html.Div(id='build-response', children='Click to build new model and train')
            ], style=col_style | {'margin-top':'20px'}),
            
            html.Div([
                html.Label(['Enter a model ID for re-training'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-train', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Re-Train', id='train-val', style={"width":"90px", "height":"30px"})
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-train', children='')
    ]),
    dcc.Tab(label="Score model", id="score-tab", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a row text (CSV) to use in scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='row-for-score', type='text', style={'width':'300px'}))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID for scoring'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-score', type='text'))
            ], style=col_style | {'margin-top':'20px'}),            
            html.Div([
                html.Button('Score', id='score-val', style={'width':'90px', "height":"30px"}),
                html.Div(id='score-response', children='Click to score')
            ], style=col_style | {'margin-top':'20px'})
        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),
        
        html.Div(id='container-button-score', children='')
    ]),

    dcc.Tab(label="Test Iris data", style=tab_style, selected_style=tab_selected_style, children=[
        html.Div([
            html.Div([
                html.Label(['Enter a dataset ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='dataset-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),
            html.Div([
                html.Label(['Enter a model ID to use in testing'], style={'font-weight': 'bold'}),
                html.Div(dcc.Input(id='model-for-test', type='text'))
            ], style=col_style | {'margin-top':'20px'}),

            html.Div([
                html.Button('Test', id='test-val'),
            ], style=col_style | {'margin-top':'20px', 'width':'90px'})

        ], style=col_style | {'margin-top':'50px', 'margin-bottom':'50px', 'width':"400px", 'border': '2px solid black'}),

        html.Div(id='container-button-test', children='')
    ])

    ])
])


'''  STARTING OF MULTI_LINE COMMENT FIELD...move code below above triple quotes to fill in and run

# callbacks for Explore data tab
'''


@app.callback(
    Output('load-response', 'children'),  
    [Input('file-for-train', 'value')],   
    [Input('load-val', 'n_clicks')]    
)

def update_output_load(file_name,nclicks):
    global df

    if nclicks != None:
        try:
            with open(file_name, "r") as data:
                # global df
                df=data.read()
                return 'The data has been uploaded'
        except Exception as ex:
            return f'Error: {str(ex)}'
    else:
        return ''


@app.callback(
    Output('build-response', 'children'),
    [Input('build-val', 'n_clicks')],
    [State('dataset-for-train', 'value')]
)
def update_output_build(nclicks,dataset_index):
    # print (nclicks)
    url = f"{BASE_URL}/model"  
    dataset = {'dataset': dataset_index}  
    if nclicks != None and dataset_index:
        # invoke new model endpoint to build and train model given data set ID
        
        try:
            response = requests.post(url, data=dataset)
            model_info = response.json()
            return f"Congratulation to succesfully built and trained model with index: {model_info.get('index')}"
        except requests.HTTPError as error:
            return f"HTTP error occurred: {error}"  
        # return the model ID 
        
    else:
        return "Enter a dataset index and click build."

@app.callback(
    Output('upload-response', 'children'),
    [Input('upload-val', 'n_clicks')],
    [State('file-for-train', 'value')]
)
def update_output_upload(nclicks, file_name):
    global df_csv

    if nclicks != None:
        # invoke the upload API endpoint
        if file_name:
            url=f'{BASE_URL}/datasets'
            files = {'train': open(file_name, 'rb')}
        # return the dataset ID generated
            try:
                response = requests.post(url, files=files)
                response.raise_for_status()  # Raise an exception for HTTP errors
                dataset_index = response.json().get('index')
                return f"Dataset uploaded successfully. Index: {dataset_index}"
            except requests.HTTPError as http_err:
                return f"HTTP error occurred: {http_err}"
            except Exception as e:
                return f"An error occurred: {e}"
        else:
            return "Please provide a filename"
    else:
        return ''

@app.callback(
# callback annotations go here
Output('indicator-graphic', 'figure'),
[Input('xaxis-column', 'value'), Input('yaxis-column', 'value')]
)
def update_graph(xaxis_column_name, yaxis_column_name):

    fig = px.scatter(x=df.loc[:,xaxis_column_name].values,
                     y=df.loc[:,yaxis_column_name].values)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=xaxis_column_name)
    fig.update_yaxes(title=yaxis_column_name)

    return fig


@app.callback(
# callback annotations go here
Output('selected_hist', 'figure'),
[Input('hist-column', 'value')]
)
def update_hist(hist_column_name):

    fig = px.histogram(df, x=hist_column_name)

    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')
    fig.update_xaxes(title=hist_column_name)

    return fig

# @app.callback(
# # callback annotations go here
# Output('datatable', 'data'),
# Output('datatable', 'columns')
# )
# def update_table():
#     return dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], page_size=15,
#         id='datatable' )
    
#     # data = df.to_dict('records')
#     # columns = [{"name": i, "id": i} for i in df.columns]
#     # return data, columns

# # callbacks for Training tab

@app.callback(
# callback annotations go here
Output('container-button-train', 'children'),
Input('train-val', 'n_clicks'),
State('model-for-train', 'value'),
State('dataset-for-train', 'value')
)
def update_output_train(nclicks, model_id, dataset_id):
    if nclicks != None:
        # add API endpoint request here
        if not model_id or not dataset_id:
            return "Model ID or Dataset ID not provided."
        url = f'{BASE_URL}/model/{model_id}'  
        data = {'dataset': dataset_id}  
        response = requests.put(url, data=data)  

        if response.status_code == 200:
            return "Successfully re-trained model."
        else:
            return f"Failed to re-train model. Status code: {response.status_code}"
        
    else:
        return "Press to Retrain"

# # callbacks for Scoring tab

@app.callback(
# callback annotations go here
Output('score-response', 'children'),
[Input('score-val', 'n_clicks')],
[State('row-for-score', 'value'),
 State('model-for-score', 'value')]
)
def update_output_score(nclicks, row, model_id):
    if nclicks:
        # add API endpoint request for scoring here with constructed input row
        if not row or not model_id:
            return "Provide model ID and input features"
        try:
            # Ensure proper input formatting
            features = ','.join(x.strip() for x in row.split(','))
            url = f'{BASE_URL}/model/{model_id}?values={features}'
            response = requests.get(url)
                
            # Check the response code
            if response.status_code == 200:
                try:
                    score_result = response.json()['result']
                    return f"The model's score result is: {score_result}"
                except (KeyError, ValueError) as e:
                    return "Failed to parse model response, please check the model's output format."
            else:
                return f"Failed to score model, server responded with status code: {response.status_code}"
            
        except requests.RequestException as e:
            return f"Network or request error occurred: {str(e)}"
        except Exception as e:
            return f"An error occurred: {str(e)}"
        else:
            return ""
    

# # callbacks for Testing tab

@app.callback(
# callback annotations go here
Output('container-button-test', 'children'),
[Input('test-val', 'n_clicks')],
[State('dataset-for-test', 'value'), State('model-for-test', 'value')]
)
def update_output_test(nclicks,dataset_id, model_id):
    if nclicks != None:
        # add API endpoint request for testing with given dataset ID
        url = f'{BASE_URL}/model/{model_id}/test?dataset={dataset_id}'
        response = requests.get(url)
        if response.status_code == 200:
            test_results = response.json()
            return f"The Model tested successfully. Results: {test_results}"
        else:
            return "Failed to test model."

    else:
        return ""



if __name__ == '__main__':
    app.run_server(debug=True)
