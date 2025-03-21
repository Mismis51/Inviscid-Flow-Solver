from dash import dcc, html

def app_layout(app, cl_figure, cd_figure, cm_figure):
    # Reusable style components
    UPLOAD_STYLE = {
        'width': '80%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '20px auto'
    }
    
    INPUT_CONTAINER_STYLE = {
        'textAlign': 'center',
        'marginBottom': '20px'
    }
    
    COEFFICIENT_BOX_STYLE = {
        'border': '2px solid #007BFF',
        'borderRadius': '10px',
        'padding': '20px',
        'maxWidth': '300px',
        'margin': '20px auto',
        'backgroundColor': '#f8f9fa'
    }
    
    GRID_STYLE = {
        'display': 'flex',
        'flexDirection': 'row',
        'gap': '20px',
        'padding': '20px'
    }

    app.layout = html.Div([
        html.H1("Interactive Flow Simulation"),
        
        # Data stores
        dcc.Store(id='geometry-vertex', storage_type='session'),
        dcc.Store(id='solver-rhs', storage_type='session'),
        dcc.Store(id='geometry-store'),
        dcc.Store(id='precomputed-aero-data'),
        dcc.Store(id='current-cp-data'),
        
        # File upload section
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Airfoil .dat File')]),
            style=UPLOAD_STYLE,
            multiple=False
        ),
        html.Div(id='upload-status', style={'textAlign': 'center', 'marginBottom': '20px'}),
        
        # Configuration controls
        html.Div([
            html.Div([
                html.Label("Number of Vertices:", style={'marginRight': '10px'}),
                dcc.Input(
                    id='nb-vertex-input',
                    type='number',
                    value=256,
                    min=4,
                    step=1,
                    style={'width': '100px', 'marginRight': '20px'}
                ),
                html.Label("Resolution:", style={'marginRight': '10px'}),
                dcc.Input(
                    id='resolution-input',
                    type='number',
                    value=200,
                    min=10,
                    step=1,
                    style={'width': '100px', 'marginRight': '20px'}
                ),
                html.Div([
                    html.Label("NACA 4-Digit:", style={'marginRight': '10px'}),
                    dcc.Input(
                        id='naca-m', type='number', min=0, max=9, step=1, placeholder='M',
                        style={'width': '60px', 'marginRight': '5px'}
                    ),
                    dcc.Input(
                        id='naca-p', type='number', min=1, max=9, step=1, placeholder='P',
                        style={'width': '60px', 'marginRight': '5px'}
                    ),
                    dcc.Input(
                        id='naca-t', type='number', min=0, max=99, step=1, placeholder='T',
                        style={'width': '80px', 'marginRight': '10px'}
                    ),
                    html.Button('Generate NACA', id='generate-naca-button', n_clicks=0,
                              style={'padding': '5px 15px'})
                ], style={'display': 'inline-block'})
            ], style=INPUT_CONTAINER_STYLE)
        ]),
        
        # Angle of attack controls
        html.Div([
            html.Div("Adjust the angle of attack:"),
            dcc.Slider(
                id='angle-slider',
                min=-15,
                max=15,
                step=0.5,
                value=0,
                marks={i: f'{i}°' for i in range(-15, 16, 5)},
                updatemode='drag'
            ),
            html.Div([
                html.Label("Custom Angle (°):", style={'display': 'inline-block', 'marginRight': '10px'}),
                dcc.Input(
                    id='angle-input',
                    type='number',
                    value=0,
                    step='any',
                    style={'width': '100px', 'display': 'inline-block'}
                )
            ], style={'marginTop': '10px'}),
        ]),
        
        # Visualization controls
        dcc.RadioItems(
            id='plot-type',
            options=[
                {'label': ' Streamlines', 'value': 'streamlines'},
                {'label': ' Velocities', 'value': 'velocities'}
            ],
            value='streamlines',
            inline=True,
            style={'margin': '20px 0'}
        ),
        
        # Main graphs
        html.Div([
            dcc.Graph(id='flow-graph'),
            dcc.Graph(id='cp-graph'),
        ], style={'display': 'flex', 'flexDirection': 'row'}),

        
        # Real-time coefficients
        html.Div([
            html.H3("Aerodynamic Coefficients", style={'textAlign': 'center'}),
            html.Div([
                html.P([html.Span("Cl: ", style={'fontWeight': 'bold', 'width': '80px'}), 
                       html.Span(id='cl-value', children='0.00')], 
                      style={'display': 'flex'}),
                html.P([html.Span("Cd: ", style={'fontWeight': 'bold', 'width': '80px'}), 
                       html.Span(id='cd-value', children='0.00')], 
                      style={'display': 'flex'}),
                html.P([html.Span("Cm_c/4: ", style={'fontWeight': 'bold', 'width': '80px'}), 
                       html.Span(id='cm-value', children='0.00')], 
                      style={'display': 'flex'})
            ], style=COEFFICIENT_BOX_STYLE)
        ]),
        
        # Precomputed graphs
        html.Div([
            html.H3("Precomputed Aerodynamic Characteristics", 
                   style={'textAlign': 'center', 'marginTop': '40px'}),
            html.Div([
                dcc.Graph(id='cl-graph', figure=cl_figure, style={'flex': 1}),
                dcc.Graph(id='cd-graph', figure=cd_figure, style={'flex': 1}),
                dcc.Graph(id='cm-graph', figure=cm_figure, style={'flex': 1})
            ], style=GRID_STYLE)
        ]),
        
        # Data export
        html.Div([
            html.Button("Download Data", id="btn-download", style={'margin': '20px'}),
            dcc.Download(id="download-data")
        ], style={'textAlign': 'center'})
    ])
