#
# Goal: plot a scatter plot with the Ames housing dataset
#
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor,SGDClassifier
import pandas
import numpy as np
from sklearn.model_selection import cross_val_score,ShuffleSplit
from sklearn.cross_validation import train_test_split

df = pandas.read_excel('AmesHousing.xls')
df1 = df.drop(['Fireplace Qu','Pool QC','Fence','Misc Feature','Alley'], 1)
df1 = df1.dropna()
df2 = pandas.get_dummies(df1, columns=['Neighborhood'])
df3 = pandas.get_dummies(df2, columns=['Sale Condition'])

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def model_selection():
    op = ['LinearRegression', 'Lasso', 'Ridge', 'Elastic Net', 'DecisionTreeRegressor', 'SGDRegressor']
    options = [ {'label': v, 'value': v} for v in op ]
    return dcc.Dropdown(
        id = 'dropdown-model',
        options = options,
        value=options[0]['value'],
        )
# make data for selected dataset
def makedataset(name):
	if name == 'ames':
		df = pandas.read_excel('AmesHousing.xls')
	else:
		df = pandas.read_csv('cali_housing.csv')

	return df



app.layout = html.Div(children=[
	html.H1(children='AmesHousing'),


    html.Div(children=[html.P(f'Select Dataset', style={'margin-left': '3px'}), dcc.Dropdown(
        id='dropdown_dataset',
        options=[
            {'label': 'Ames_housing', 'value': 'ames'},
            {'label': 'Cali_housing', 'value': 'ca'},
        ],
        value= 'ames',
        clearable=False,
        searchable=False
    )]),

    html.Div(children=[html.P(f'Neighborhood', style={'margin-left': '3px'}),
        dcc.Dropdown(
            id='neighbour',
            options=[{'label': i, 'value': i} for i in df.Neighborhood.unique() ],
            value='Neighborhood'
        ),],
 		style={'width': '48%'},
    ),

    html.Div(children=[html.P(f'Sale Condition', style={'margin-left': '3px'}),
        dcc.Dropdown(
            id='condition',
            options=[{'label': i, 'value': i} for i in df['Sale Condition'].unique() ],
            value=''
        ),],
    	style={'width': '48%', 'display': 'inline-block'},
    ),

    html.H3(children='Cross Validation'),

    dcc.Checklist(
    id='feature_list',
    options=[
        {'label': 'Overall Qualility (1 - 10)', 'value': 'Overall Qual'},
        {'label': 'TotRms AbvGrd', 'value': 'TotRms AbvGrd'},
        {'label': 'Full Bath', 'value': 'Full Bath'},
        {'label': 'Gr Liv Area', 'value': 'Gr Liv Area'},
        {'label': 'Bedroom AbvGr', 'value': 'Bedroom AbvGr'},
        {'label': 'Lot Area', 'value': 'Lot Area'},
        {'label': 'Year Built', 'value': 'Year Built'},
        {'label': 'Garage Area', 'value': 'Garage Area'},
        {'label': 'Garage Cars', 'value': 'Garage Cars'}
    ],
    values=['Overall Qual'],
    labelStyle={'display': 'inline-block'}
	),

	html.Div(children=[html.P(f'Select Model', style={'margin-left': '3px'}), model_selection() ],
        # Step 5
        style={'width': '48%', 'display': 'inline-block'},
    ),

    html.Div(id='model_output'),

    # html.H3(children='Price Prediction'),
	dcc.Markdown('''### Price Prediction'''),

	html.Div(children=[

    html.Div(children = [ html.P(f'Overall Qualility (1 - 10)', style={'margin-left': '3px'}), dcc.Input(id='text-input1', value='', type='number')
    	], style={'width': '30%', 'display': 'inline-block'}),
    html.Div(children = [ html.P(f'Total Rooms', style={'margin-left': '3px'}), dcc.Input(id='text-input2', value='', type='number')
    	],style={'width': '30%', 'display': 'inline-block'}),
    html.Div(children = [ html.P(f'Total Bedrooms', style={'margin-left': '3px'}), dcc.Input(id='text-input3', value='', type='number')
    	],style={'width': '30%', 'display': 'inline-block'}),
    html.Div(children = [html.P(f'Garage Cars', style={'margin-left': '3px'}), dcc.Input(id='text-input4', value='', type='number'),
    	],style={'width': '30%', 'display': 'inline-block'}),
    html.Div( children = [html.P(f'Lot Area (size in square feet)', style={'margin-left': '3px'}), dcc.Input(id='text-input5', value='', type='number'),
    	],style={'width': '30%', 'display': 'inline-block'}),
    html.Div(children = [html.P(f'Year Built', style={'margin-left': '3px'}), dcc.Input(id='text-input6', value='', type='number' ),
    	],style={'width': '30%', 'display': 'inline-block'}),
	],
	style={'display': 'inline-block'},),

	html.Div(id='predict_output'),

	dcc.Graph(id='graph'),

	dcc.Graph(
        id='hist_graph',
        figure={
            'data': [
                {
                    'x': df3['SalePrice'],       
                    'mode': 'markers',
                    'type': 'histogram'
                },
            ],
            'layout': {
            'title': 'AmesHousing SalePrice Histogram'
            }
        })
])


@app.callback(
	Output('graph', 'figure'),
	[Input('neighbour', 'value'),
	Input('condition', 'value')])
def update_graph(neighbour_name, condition):
	data = df[ df.Neighborhood == neighbour_name ]
	data1 = data[df['Sale Condition'] == condition]
	return dict(
		data = [go.Scatter(
			x = data1.SalePrice,
			y = data1['Gr Liv Area'],
			mode = 'markers',
			name = neighbour_name,
		)],
		layout = go.Layout(
			title = 'Ames dataset',
			xaxis={'title': 'SalePrice'},
			yaxis={'title': 'Gr Liv Area'},
		),
	)

@app.callback(
	Output(component_id='model_output', component_property='children'),
	[
		Input(component_id='feature_list', component_property='values'),
		Input(component_id='dropdown-model', component_property='value')
	]
)
def update_output_div(feature_values,model_value):

	print(type(feature_values))
	neighbor = ['Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr','Neighborhood_Timber', 'Neighborhood_Veenker']
	features= feature_values + neighbor

	print(features)
	X = df3[features]

	y = df3.SalePrice

	if model_value == 'Lasso':
		model = Lasso()
	elif model_value == 'Ridge':
		model = Ridge()
	elif model_value == 'Elastic Net':
		model = ElasticNet()
	elif model_value == 'DecisionTreeRegressor':
		model = DecisionTreeRegressor()
	elif model_value == 'SGDRegressor':
		model = SGDRegressor(max_iter=10)
	else:
		model = LinearRegression()

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


	v = ShuffleSplit(n_splits=100, test_size=0.1)
	scores = cross_val_score(model, X_train, y_train, cv = v)

	score = np.around(sum(scores)/len(scores),2)

	# return html.H4('The predict house price is {}'.format(price))
	return html.H3(children =['Cross validation score is {}'.format(score)], style={'color': 'green',})

@app.callback(
	Output(component_id='predict_output', component_property='children'),
	[
		Input(component_id='text-input1', component_property='value'),
		Input(component_id='text-input2', component_property='value'),
		Input(component_id='text-input3', component_property='value'),
		Input(component_id='text-input4', component_property='value'),
		Input(component_id='text-input5', component_property='value'),
		Input(component_id='text-input6', component_property='value')
	]
)
def update_output_div(input1, input2, input3, input4,input5,input6):


	X = df3[ ['Overall Qual','TotRms AbvGrd', 'Bedroom AbvGr', 'Garage Cars', 'Lot Area','Year Built']]
	y = df3.SalePrice

	# model = LinearRegression()
	model = Ridge()
	# model.fit(X_train, y_train)
	model.fit(X, y)

	p = model.predict([[input1,input2, input3,input4, input5, input6]])


	price = np.around(p,2)


	return html.H3(children =['The predict house price is {}'.format(price)], style ={'color': 'green'})



if __name__ == '__main__':
	app.run_server(debug=True)