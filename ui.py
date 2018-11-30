#
# Goal: plot a scatter plot with the iris dataset
#
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor,SGDClassifier
import pandas

df = pandas.read_excel('AmesHousing.xls')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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

    html.Div(id='output_ames', children = [

	    html.H3(children='Price Prediction'),

	    html.Div(className='row', children=[

		    html.Div(className='four columns',children = [ html.P(f'Overall Qualility (1 - 10)', style={'margin-left': '3px'}), dcc.Input(id='text-input1', value='', type='number', placeholder='Overall Qualility')
		    	]),
		    html.Div(className='four columns', children = [html.P(f'Lot Area', style={'margin-left': '3px'}), dcc.Input(id='text-input2', value='', type='number', placeholder='Lot Area'),
		    	]),
		    html.Div(className='four columns',children = [html.P(f'Year Built', style={'margin-left': '3px'}), dcc.Input(id='text-input3', value='', type='number', placeholder='Year Built' ),
		    	]),
    		]),

	    html.P(f'Neighborhood', style={'margin-left': '3px'}), 
   	 	html.Div([
        dcc.Dropdown(
            id='neighbour',
            options=[{'label': i, 'value': i} for i in df.Neighborhood.unique() ],
            value='Neighborhood'
        )]),

    	]),

	# html.Div(children='Neighborhood'),
 #    html.Div([
 #        dcc.Dropdown(
 #            id='neighbour',
 #            options=[{'label': i, 'value': i} for i in df.Neighborhood.unique() ],
 #            value='Neighborhood'
 #        ),
 #    	],
 #    ),

 #    html.H3(children='Price Prediction'),

 #    html.Div(className='row', children=[

	#     html.Div(className='four columns',children = [ html.P(f'Overall Qualility (1 - 10)', style={'margin-left': '3px'}), dcc.Input(id='text-input1', value='', type='number', placeholder='Overall Qualility')
	#     	]),
	#     html.Div(className='four columns', children = [html.P(f'Lot Area', style={'margin-left': '3px'}), dcc.Input(id='text-input2', value='', type='number', placeholder='Lot Area'),
	#     	]),
	#     html.Div(className='four columns',children = [html.P(f'Year Built', style={'margin-left': '3px'}), dcc.Input(id='text-input3', value='', type='number', placeholder='Year Built' ),
	#     	]),
 #    ]),

    html.Div(id='predict_output'),

	dcc.Graph(id='graph')
])


@app.callback(
	Output(component_id='output_ames', component_property='children'),
	[Input(component_id='dropdown_dataset', component_property='value'),])
def update_div(dataset):
	print(dataset)
	if dataset == 'ames':
		print('testestestets')
		return 'You have entered {}'.to_json()
	else:
		return 'You have entered {}'




@app.callback(
	Output('graph', 'figure'),
	[Input('neighbour', 'value'),])
def update_graph(neighbour_name):
	data = df[ df.Neighborhood == neighbour_name ]
	return dict(
		data = [go.Scatter(
			x = data.SalePrice,
			y = data['Gr Liv Area'],
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
	Output(component_id='predict_output', component_property='children'),
	[
		Input(component_id='dropdown_dataset', component_property='value'),
		Input(component_id='text-input1', component_property='value'),
		Input(component_id='text-input2', component_property='value'),
		Input(component_id='text-input3', component_property='value')
	]
)
def update_output_div(dataset, input1, input2, input3):

	if dataset == 'ames':
		df = pandas.read_excel('AmesHousing.xls')

		X = df[ ['Overall Qual', 'Lot Area','Year Built']]
		y = df.SalePrice

		# model = LinearRegression()
		model = Ridge()
		# model.fit(X_train, y_train)
		model.fit(X, y)

		# p = model.predict([[9,50, 2000]])
		p = model.predict([[input1,input2, input3]])
	else:
		df = pandas.read_csv('cali_housing.csv')

		X = df[ ['housing_median_age', 'total_rooms','median_income']]
		y = df.median_house_value

		# model = LinearRegression()
		model = Ridge()
		# model.fit(X_train, y_train)
		model.fit(X, y)

		# p = model.predict([[9,50, 2000]])
		p = model.predict([[input1,input2, input3]])




	return html.H4('The predict house price is  {}'.format(p))

if __name__ == '__main__':
	app.run_server(debug=True)