import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import time

# Initialize the Dash app
app = dash.Dash(__name__)

# SIR Model Functions
def sir_model(N, I0, beta, gamma, days):
    """
    SIR model implementation using Euler's method
    """
    # Initialize arrays
    S = np.zeros(days)
    I = np.zeros(days)
    R = np.zeros(days)
    
    # Initial conditions
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    # Time step (1 day)
    dt = 1.0
    
    # Euler's method implementation
    for t in range(1, days):
        # Calculate rates
        infection_rate = beta * S[t-1] * I[t-1] / N
        recovery_rate = gamma * I[t-1]
        
        # Update populations using Euler's method
        S[t] = S[t-1] - infection_rate * dt
        I[t] = I[t-1] + infection_rate * dt - recovery_rate * dt
        R[t] = R[t-1] + recovery_rate * dt
        
        # Ensure non-negative values
        S[t] = max(0, S[t])
        I[t] = max(0, I[t])
        R[t] = max(0, R[t])
    
    return S, I, R

def calculate_metrics(S, I, R, beta, gamma):
    """
    Calculate key epidemiological metrics
    """
    # Basic reproduction number
    R0 = beta / gamma
    
    # Peak infection values
    peak_infected = np.max(I)
    peak_day = np.argmax(I)
    
    # Final recovered (total who got infected)
    final_recovered = R[-1]
    
    return {
        'R0': round(R0, 2),
        'peak_infected': int(peak_infected),
        'peak_day': int(peak_day),
        'final_recovered': int(final_recovered)
    }

def generate_people_positions(N, I0, canvas_width=600, canvas_height=400):
    """
    Generate initial positions and states for people in the simulation
    """
    people = []
    for i in range(N):
        person = {
            'id': i,
            'x': np.random.uniform(10, canvas_width - 10),
            'y': np.random.uniform(10, canvas_height - 10),
            'vx': np.random.uniform(-2, 2),
            'vy': np.random.uniform(-2, 2),
            'status': 'infected' if i < I0 else 'susceptible',
            'infection_day': 0 if i < I0 else -1,
            'radius': 4
        }
        people.append(person)
    return people

def create_animated_graph_figure(time_points, S, I, R, current_day, metrics=None):
    """
    Create animated graph figure showing progression up to current_day
    """
    # Only show data up to current day
    current_time = time_points[:current_day + 1]
    current_S = S[:current_day + 1]
    current_I = I[:current_day + 1]
    current_R = R[:current_day + 1]
    
    fig = go.Figure()
    
    # Add traces for current data
    fig.add_trace(go.Scatter(
        x=current_time, y=current_S, mode='lines', name='Susceptible (S)',
        line=dict(color='#3498db', width=3),
        hovertemplate='Day %{x}<br>Susceptible: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=current_time, y=current_I, mode='lines', name='Infected (I)',
        line=dict(color='#e74c3c', width=3),
        hovertemplate='Day %{x}<br>Infected: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=current_time, y=current_R, mode='lines', name='Recovered (R)',
        line=dict(color='#27ae60', width=3),
        hovertemplate='Day %{x}<br>Recovered: %{y:,.0f}<extra></extra>'
    ))
    
    # Add current day marker
    if current_day > 0:
        fig.add_trace(go.Scatter(
            x=[current_day], y=[current_S[-1]],
            mode='markers', name='Current Day (S)',
            marker=dict(color='#3498db', size=8, symbol='circle'),
            showlegend=False,
            hovertemplate='Day %{x}<br>Susceptible: %{y:,.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[current_day], y=[current_I[-1]],
            mode='markers', name='Current Day (I)',
            marker=dict(color='#e74c3c', size=8, symbol='circle'),
            showlegend=False,
            hovertemplate='Day %{x}<br>Infected: %{y:,.0f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[current_day], y=[current_R[-1]],
            mode='markers', name='Current Day (R)',
            marker=dict(color='#27ae60', size=8, symbol='circle'),
            showlegend=False,
            hovertemplate='Day %{x}<br>Recovered: %{y:,.0f}<extra></extra>'
        ))
    
    # Add peak infection marker if we've reached it and have metrics
    if metrics and current_day >= metrics['peak_day']:
        fig.add_trace(go.Scatter(
            x=[metrics['peak_day']], y=[metrics['peak_infected']],
            mode='markers', name='Peak Infection',
            marker=dict(color='red', size=10, symbol='star'),
            hovertemplate='Peak Day: %{x}<br>Peak Infected: %{y:,.0f}<extra></extra>'
        ))
    
    # Update layout
    max_pop = max(S[0], np.max(I), np.max(R)) * 1.1
    fig.update_layout(
        title=f'SIR Model - Population Over Time (Day {current_day})',
        xaxis={'title': 'Days', 'range': [0, len(time_points)]},
        yaxis={'title': 'Number of People', 'range': [0, max_pop]},
        hovermode='x unified',
        height=500,
        legend=dict(x=0.7, y=0.95),
        plot_bgcolor='white',
        paper_bgcolor='white',
        annotations=[
            dict(
                x=0.02, y=0.98,
                xref='paper', yref='paper',
                text=f'<b>Day {current_day}</b>',
                showarrow=False,
                font=dict(size=16, color='#2c3e50'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#2c3e50',
                borderwidth=1
            )
        ]
    )
    
    return fig

# Define the layout
app.layout = html.Div([
    # Main title
    html.H1("SIR Epidemic Transmission Model", 
            style={'text-align': 'center', 'margin-bottom': '30px', 'color': '#2c3e50'}),
    
    # Simulation Mode Selector
    html.Div([
        html.H3("Simulation Mode:", style={'color': '#34495e', 'margin-bottom': '10px'}),
        dcc.RadioItems(
            id='simulation-mode',
            options=[
                {'label': 'Static Graph (Complete Model)', 'value': 'static'},
                {'label': 'Animated Graph (Progressive Build)', 'value': 'animated_graph'},
                {'label': 'People Simulation (Moving Dots)', 'value': 'animated_people'}
            ],
            value='static',
            labelStyle={'display': 'block', 'margin-bottom': '10px'},
            style={'margin-bottom': '20px'}
        )
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    
    # Main container
    html.Div([
        # Left panel - Controls and inputs
        html.Div([
            # Input Parameters Section
            html.H3("Model Parameters", style={'color': '#34495e', 'margin-bottom': '20px'}),
            
            # Total Population
            html.Div([
                html.Label("Total Population (N):", style={'font-weight': 'bold'}),
                dcc.Input(
                    id='population',
                    type='number',
                    value=200,
                    min=50,
                    max=1000,
                    style={'width': '100%', 'padding': '8px', 'margin-top': '5px'}
                ),
                html.Small("Total number of people in the population", 
                          style={'color': '#7f8c8d', 'font-style': 'italic'})
            ], style={'margin-bottom': '20px'}),
            
            # Initial Infected
            html.Div([
                html.Label("Initial Infected (I₀):", style={'font-weight': 'bold'}),
                dcc.Input(
                    id='initial-infected',
                    type='number',
                    value=5,
                    min=1,
                    max=50,
                    style={'width': '100%', 'padding': '8px', 'margin-top': '5px'}
                ),
                html.Small("Number of infected people at start", 
                          style={'color': '#7f8c8d', 'font-style': 'italic'})
            ], style={'margin-bottom': '20px'}),
            
            # Transmission Rate (Beta)
            html.Div([
                html.Label("Transmission Rate (β):", style={'font-weight': 'bold'}),
                dcc.Input(
                    id='beta',
                    type='number',
                    value=0.3,
                    min=0.01,
                    max=2.0,
                    step=0.01,
                    style={'width': '100%', 'padding': '8px', 'margin-top': '5px'}
                ),
                html.Small("Rate of disease transmission per contact", 
                          style={'color': '#7f8c8d', 'font-style': 'italic'})
            ], style={'margin-bottom': '20px'}),
            
            # Recovery Rate (Gamma)
            html.Div([
                html.Label("Recovery Rate (γ):", style={'font-weight': 'bold'}),
                dcc.Input(
                    id='gamma',
                    type='number',
                    value=0.1,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    style={'width': '100%', 'padding': '8px', 'margin-top': '5px'}
                ),
                html.Small("Rate at which infected people recover", 
                          style={'color': '#7f8c8d', 'font-style': 'italic'})
            ], style={'margin-bottom': '20px'}),
            
            # Simulation Duration
            html.Div([
                html.Label("Simulation Duration (days):", style={'font-weight': 'bold'}),
                dcc.Input(
                    id='duration',
                    type='number',
                    value=100,
                    min=30,
                    max=365,
                    style={'width': '100%', 'padding': '8px', 'margin-top': '5px'}
                ),
                html.Small("How many days to simulate", 
                          style={'color': '#7f8c8d', 'font-style': 'italic'})
            ], style={'margin-bottom': '20px'}),
            
            # Animation Speed (for animated modes)
            html.Div([
                html.Label("Animation Speed:", style={'font-weight': 'bold'}),
                dcc.Slider(
                    id='animation-speed',
                    min=1,
                    max=10,
                    step=1,
                    value=5,
                    marks={i: str(i) for i in range(1, 11)},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Small("Speed of animation (1=slow, 10=fast)", 
                          style={'color': '#7f8c8d', 'font-style': 'italic'})
            ], style={'margin-bottom': '30px'}, id='speed-control'),
            
            # Control Buttons
            html.H3("Simulation Controls", style={'color': '#34495e', 'margin-bottom': '20px'}),
            html.Div([
                html.Button('Start Simulation', id='start-btn', n_clicks=0,
                           style={'background-color': '#27ae60', 'color': 'white', 
                                  'border': 'none', 'padding': '12px 24px', 
                                  'margin-right': '10px', 'border-radius': '5px',
                                  'cursor': 'pointer', 'font-size': '14px'}),
                
                html.Button('Pause', id='pause-btn', n_clicks=0,
                           style={'background-color': '#f39c12', 'color': 'white', 
                                  'border': 'none', 'padding': '12px 24px', 
                                  'margin-right': '10px', 'border-radius': '5px',
                                  'cursor': 'pointer', 'font-size': '14px'}),
                
                html.Button('Reset', id='reset-btn', n_clicks=0,
                           style={'background-color': '#e74c3c', 'color': 'white', 
                                  'border': 'none', 'padding': '12px 24px', 
                                  'border-radius': '5px', 'cursor': 'pointer', 
                                  'font-size': '14px'})
            ], style={'margin-bottom': '30px'}),
            
            # Information Panel
            html.H3("Model Information", style={'color': '#34495e', 'margin-bottom': '20px'}),
            html.Div([
                html.H4("SIR Model Assumptions:", style={'color': '#2c3e50'}),
                html.Ul([
                    html.Li("Population is closed (no births/deaths)"),
                    html.Li("Homogeneous mixing (everyone can contact everyone)"),
                    html.Li("Recovery provides permanent immunity"),
                    html.Li("Infection period follows exponential distribution")
                ], style={'color': '#34495e', 'line-height': '1.6'}),
                
                html.H4("Key Metrics:", style={'color': '#2c3e50', 'margin-top': '20px'}),
                html.Div(id='key-metrics', children=[
                    html.P("R₀ (Basic Reproduction Number): --", style={'margin': '5px 0'}),
                    html.P("Peak Infections: --", style={'margin': '5px 0'}),
                    html.P("Peak Day: --", style={'margin': '5px 0'}),
                    html.P("Final Recovered: --", style={'margin': '5px 0'})
                ], style={'background-color': '#ecf0f1', 'padding': '15px', 
                         'border-radius': '5px', 'color': '#2c3e50'})
            ])
            
        ], style={'width': '30%', 'padding': '20px', 'background-color': '#f8f9fa',
                  'border-radius': '10px', 'margin-right': '20px'}),
        # Right panel - Visualization
        html.Div([
            # Day counter for animated modes
            html.Div(id='day-counter', children="Day: 0", 
                    style={'text-align': 'center', 'font-size': '24px', 'font-weight': 'bold',
                           'color': '#2c3e50', 'margin-bottom': '20px', 'display': 'none'}),
            
            # Population stats for people simulation mode
            html.Div(id='population-stats', children=[
                html.Div([
                    html.Div("Susceptible", style={'color': 'white', 'font-weight': 'bold'}),
                    html.Div("0", id='susceptible-count', style={'font-size': '18px'})
                ], style={'background': '#3498db', 'padding': '10px', 'border-radius': '5px', 
                         'text-align': 'center', 'color': 'white', 'margin': '5px'}),
                
                html.Div([
                    html.Div("Infected", style={'color': 'white', 'font-weight': 'bold'}),
                    html.Div("0", id='infected-count', style={'font-size': '18px'})
                ], style={'background': '#e74c3c', 'padding': '10px', 'border-radius': '5px', 
                         'text-align': 'center', 'color': 'white', 'margin': '5px'}),
                
                html.Div([
                    html.Div("Recovered", style={'color': 'white', 'font-weight': 'bold'}),
                    html.Div("0", id='recovered-count', style={'font-size': '18px'})
                ], style={'background': '#27ae60', 'padding': '10px', 'border-radius': '5px', 
                         'text-align': 'center', 'color': 'white', 'margin': '5px'})
            ], style={'display': 'none', 'justify-content': 'space-around', 'margin-bottom': '20px'}),
            
            html.H3("Epidemic Progression", id='visualization-title',
                   style={'text-align': 'center', 'color': '#34495e', 'margin-bottom': '20px'}),
            
            # Graph visualization (for static and animated graph modes)
            dcc.Graph(
                id='sir-graph',
                figure={
                    'data': [
                        go.Scatter(x=[0], y=[200], mode='lines', name='Susceptible (S)', 
                                  line=dict(color='#3498db', width=3)),
                        go.Scatter(x=[0], y=[5], mode='lines', name='Infected (I)', 
                                  line=dict(color='#e74c3c', width=3)),
                        go.Scatter(x=[0], y=[0], mode='lines', name='Recovered (R)', 
                                  line=dict(color='#27ae60', width=3))
                    ],
                    'layout': go.Layout(
                        title='SIR Model - Population Over Time',
                        xaxis={'title': 'Days'},
                        yaxis={'title': 'Number of People'},
                        hovermode='x unified',
                        template='plotly_white',
                        height=500
                    )
                }
            ),
            
            # Canvas for people simulation (hidden by default)
            html.Canvas(
                id='animation-canvas',
                width=600,
                height=400,
                style={'border': '2px solid #34495e', 'border-radius': '8px',
                       'background': 'white', 'display': 'none', 'margin': '0 auto'}
            ),
            
            # Status indicator
            html.Div(id='simulation-status', 
                    children="Ready to simulate", 
                    style={'text-align': 'center', 'margin-top': '20px', 
                           'font-size': '16px', 'color': '#7f8c8d'}),
            
            # Animation legend (for people simulation mode)
            html.Div([
                html.Div([
                    html.Div(style={'width': '12px', 'height': '12px', 'border-radius': '50%',
                                   'background': '#3498db', 'margin-right': '5px'}),
                    html.Span("Susceptible")
                ], style={'display': 'flex', 'align-items': 'center', 'margin': '0 10px'}),
                
                html.Div([
                    html.Div(style={'width': '12px', 'height': '12px', 'border-radius': '50%',
                                   'background': '#e74c3c', 'margin-right': '5px'}),
                    html.Span("Infected")
                ], style={'display': 'flex', 'align-items': 'center', 'margin': '0 10px'}),
                
                html.Div([
                    html.Div(style={'width': '12px', 'height': '12px', 'border-radius': '50%',
                                   'background': '#27ae60', 'margin-right': '5px'}),
                    html.Span("Recovered")
                ], style={'display': 'flex', 'align-items': 'center', 'margin': '0 10px'})
            ], id='animation-legend', style={'display': 'none', 'justify-content': 'center', 
                                           'margin-top': '10px'}),
            
            # Hidden div to store animation data
            html.Div(id='animation-data', style={'display': 'none'}),
            html.Div(id='graph-animation-data', style={'display': 'none'}),
            
            # Interval components for different animation types
            dcc.Interval(
                id='people-animation-interval',
                interval=50,  # 50ms for smooth people movement
                n_intervals=0,
                disabled=True
            ),
            
            dcc.Interval(
                id='graph-animation-interval',
                interval=100,  # 100ms for graph animation
                n_intervals=0,
                disabled=True
            )
            
        ], style={'width': '65%', 'padding': '20px'})
        
    ], style={'display': 'flex', 'max-width': '1400px', 'margin': '0 auto'})
    
], style={'padding': '20px', 'font-family': 'Arial, sans-serif'})

# Callback to toggle visibility based on simulation mode
@app.callback(
    [Output('sir-graph', 'style'),
     Output('animation-canvas', 'style'),
     Output('day-counter', 'style'),
     Output('population-stats', 'style'),
     Output('animation-legend', 'style'),
     Output('speed-control', 'style'),
     Output('population', 'max'),
     Output('visualization-title', 'children')],
    [Input('simulation-mode', 'value')],
    prevent_initial_call=True
    
)
def toggle_simulation_mode(mode):
    if mode == 'animated_people':
        return (
            {'display': 'none'},  # Hide graph
            {'border': '2px solid #34495e', 'border-radius': '8px', 
             'background': 'white', 'display': 'block', 'margin': '0 auto'},  # Show canvas
            {'text-align': 'center', 'font-size': '24px', 'font-weight': 'bold',
             'color': '#2c3e50', 'margin-bottom': '20px', 'display': 'block'},  # Show day counter
            {'display': 'flex', 'justify-content': 'space-around', 'margin-bottom': '20px'},  # Show stats
            {'display': 'flex', 'justify-content': 'center', 'margin-top': '10px'},  # Show legend
            {'margin-bottom': '30px'},  # Show speed control
            500,  # Max population for people simulation
            "People Simulation"
        )
    elif mode == 'animated_graph':
        return (
            {},  # Show graph
            {'display': 'none'},  # Hide canvas
            {'display': 'none'},  # Hide day counter (shown in graph title)
            {'display': 'none'},  # Hide stats
            {'display': 'none'},  # Hide legend
            {'margin-bottom': '30px'},  # Show speed control
            1000000,  # Max population for graph mode
            "Animated Graph"
        )
    else:  # static mode
        return (
            {},  # Show graph
            {'display': 'none'},  # Hide canvas
            {'display': 'none'},  # Hide day counter
            {'display': 'none'},  # Hide stats
            {'display': 'none'},  # Hide legend
            {'display': 'none'},  # Hide speed control
            1000000,  # Max population for graph mode
            "Static Graph"
        )
# Main callback for simulation
@app.callback(
    [Output('sir-graph', 'figure'),
     Output('key-metrics', 'children'),
     Output('simulation-status', 'children', allow_duplicate=True),  # Add allow_duplicate=True here
     Output('animation-data', 'children'),
     Output('graph-animation-data', 'children'),
     Output('people-animation-interval', 'disabled'),
     Output('graph-animation-interval', 'disabled')],
    [Input('start-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks'),
     Input('pause-btn', 'n_clicks'),
     Input('simulation-mode', 'value')],
    [State('population', 'value'),
     State('initial-infected', 'value'),
     State('beta', 'value'),
     State('gamma', 'value'),
     State('duration', 'value')],
    prevent_initial_call=True

)
def update_simulation(start_clicks, reset_clicks, pause_clicks, mode, N, I0, beta, gamma, days):
    ctx = callback_context
    if not ctx.triggered:
        button_id = 'start-btn'
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Validate inputs
    if not all([N, I0, beta, gamma, days]) or any(x <= 0 for x in [N, I0, beta, gamma, days]):
        empty_fig = create_empty_figure(N or 200, I0 or 5)
        return empty_fig, create_empty_metrics(), "Invalid inputs", "", "", True, True
    
    # Reset button clicked
    if button_id == 'reset-btn' and reset_clicks > 0:
        initial_fig = create_empty_figure(N, I0)
        initial_metrics = create_empty_metrics()
        return initial_fig, initial_metrics, "Reset complete - Ready to simulate", "", "", True, True
    
    # Pause button clicked
    if button_id == 'pause-btn' and pause_clicks > 0:
        return dash.no_update, dash.no_update, "Simulation paused", dash.no_update, dash.no_update, True, True
    
    # Start simulation
    if button_id == 'start-btn' and start_clicks > 0:
        if mode == 'animated_people':
            # Initialize people animation data
            people = generate_people_positions(N, I0)
            animation_data = {
                'people': people,
                'day': 0,
                'frame': 0,
                'running': True,
                'parameters': {'N': N, 'I0': I0, 'beta': beta, 'gamma': gamma, 'days': days}
            }
            return dash.no_update, dash.no_update, "People animation started", json.dumps(animation_data), "", False, True
        
        elif mode == 'animated_graph':
            # Initialize graph animation data
            S, I, R = sir_model(N, I0, beta, gamma, days)
            time_points = np.arange(days)
            metrics = calculate_metrics(S, I, R, beta, gamma)
            
            graph_data = {
                'S': S.tolist(),
                'I': I.tolist(),
                'R': R.tolist(),
                'time_points': time_points.tolist(),
                'current_day': 0,
                'metrics': metrics,
                'running': True
            }
            
            # Create initial figure with just day 0
            initial_fig = create_animated_graph_figure(time_points, S, I, R, 0, metrics)
            metrics_display = create_metrics_display(metrics)
            
            return initial_fig, metrics_display, "Graph animation started", "", json.dumps(graph_data), True, False
        
        else:  # static mode
            # Run complete simulation and show final result
            S, I, R = sir_model(N, I0, beta, gamma, days)
            time_points = np.arange(days)
            metrics = calculate_metrics(S, I, R, beta, gamma)
            
            # Create static figure with complete data
            fig = create_static_figure(time_points, S, I, R, metrics)
            metrics_display = create_metrics_display(metrics)
            
            return fig, metrics_display, "Static simulation complete", "", "", True, True
    
    # Default case
    initial_fig = create_empty_figure(N, I0)
    return initial_fig, create_empty_metrics(), "Ready to simulate", "", "", True, True

# Callback for graph animation updates
# Callback for graph animation updates
# Callback for graph animation updates
@app.callback(
    [Output('sir-graph', 'figure', allow_duplicate=True),
     Output('graph-animation-data', 'children', allow_duplicate=True),
     Output('simulation-status', 'children', allow_duplicate=True)],
    [Input('graph-animation-interval', 'n_intervals')],
    [State('graph-animation-data', 'children'),
     State('animation-speed', 'value')],
    prevent_initial_call=True
)
def update_graph_animation(n_intervals, graph_data_json, speed):
    if not graph_data_json:
        return dash.no_update, dash.no_update, dash.no_update

    try:
        graph_data = json.loads(graph_data_json)
        if not graph_data.get('running', False):
            return dash.no_update, dash.no_update, dash.no_update

        # Calculate day increment based on speed
        day_increment = max(1, speed // 2)
        current_day = graph_data['current_day']
        new_day = min(current_day + day_increment, len(graph_data['time_points']) - 1)

        # Update current day
        graph_data['current_day'] = new_day

        # Create updated figure
        S = np.array(graph_data['S'])
        I = np.array(graph_data['I'])
        R = np.array(graph_data['R'])
        time_points = np.array(graph_data['time_points'])
        metrics = graph_data['metrics']

        fig = create_animated_graph_figure(time_points, S, I, R, new_day, metrics)

        # Check if animation is complete
        if new_day >= len(time_points) - 1:
            graph_data['running'] = False
            status = f"Graph animation complete - Final day: {new_day}"
        else:
            status = f"Animating graph - Day: {new_day}"

        return fig, json.dumps(graph_data), status

    except Exception as e:
        return dash.no_update, dash.no_update, f"Animation error: {str(e)}"

# Add this callback for people animation updates
@app.callback(
    [Output('day-counter', 'children'),
     Output('susceptible-count', 'children'),
     Output('infected-count', 'children'),
     Output('recovered-count', 'children'),
     Output('animation-data', 'children', allow_duplicate=True),
     Output('simulation-status', 'children', allow_duplicate=True),
     Output('people-animation-interval', 'disabled', allow_duplicate=True)],
    [Input('people-animation-interval', 'n_intervals')],
    [State('animation-data', 'children'),
     State('animation-speed', 'value')],
    prevent_initial_call=True
)
def update_people_animation(n_intervals, animation_data_json, speed):
    if not animation_data_json:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    try:
        animation_data = json.loads(animation_data_json)
        if not animation_data.get('running', False):
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        people = animation_data['people']
        params = animation_data['parameters']
        current_frame = animation_data['frame']
        current_day = animation_data['day']
        
        # Update frame counter based on speed
        frames_per_day = max(20, 100 - speed * 8)  # Faster speed = fewer frames per day
        animation_data['frame'] = current_frame + 1
        
        # Check if we should advance to next day
        if current_frame % frames_per_day == 0 and current_frame > 0:
            animation_data['day'] = current_day + 1
            current_day = animation_data['day']
        
        # Stop if we've reached the simulation duration
        if current_day >= params['days']:
            animation_data['running'] = False
            # Count final populations
            susceptible = len([p for p in people if p['status'] == 'susceptible'])
            infected = len([p for p in people if p['status'] == 'infected'])
            recovered = len([p for p in people if p['status'] == 'recovered'])
            
            return (f"Day: {current_day} (Complete)", 
                    str(susceptible), str(infected), str(recovered),
                    json.dumps(animation_data), 
                    "People simulation complete", True)
        
        # Update people positions and interactions
        canvas_width, canvas_height = 600, 400
        infection_radius = 15
        
        # Move people
        for person in people:
            # Update position
            person['x'] += person['vx']
            person['y'] += person['vy']
            
            # Bounce off walls
            if person['x'] <= person['radius'] or person['x'] >= canvas_width - person['radius']:
                person['vx'] = -person['vx']
                person['x'] = max(person['radius'], min(canvas_width - person['radius'], person['x']))
            
            if person['y'] <= person['radius'] or person['y'] >= canvas_height - person['radius']:
                person['vy'] = -person['vy']
                person['y'] = max(person['radius'], min(canvas_height - person['radius'], person['y']))
        
        # Handle infections (only on day boundaries)
        if current_frame % frames_per_day == 0 and current_frame > 0:
            # Check for new infections
            infected_people = [p for p in people if p['status'] == 'infected']
            susceptible_people = [p for p in people if p['status'] == 'susceptible']
            
            for infected_person in infected_people:
                for susceptible_person in susceptible_people:
                    # Calculate distance
                    dx = infected_person['x'] - susceptible_person['x']
                    dy = infected_person['y'] - susceptible_person['y']
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Check if infection occurs
                    if distance < infection_radius:
                        # Probability of infection based on beta
                        if np.random.random() < params['beta']:
                            susceptible_person['status'] = 'infected'
                            susceptible_person['infection_day'] = current_day
            
            # Handle recoveries
            for person in people:
                if person['status'] == 'infected':
                    days_infected = current_day - person['infection_day']
                    # Probability of recovery based on gamma
                    if np.random.random() < params['gamma']:
                        person['status'] = 'recovered'
        
        # Update animation data
        animation_data['people'] = people
        
        # Count current populations
        susceptible = len([p for p in people if p['status'] == 'susceptible'])
        infected = len([p for p in people if p['status'] == 'infected'])
        recovered = len([p for p in people if p['status'] == 'recovered'])
        
        return (f"Day: {current_day}", 
                str(susceptible), str(infected), str(recovered),
                json.dumps(animation_data), 
                f"People simulation running - Day {current_day}", False)
        
    except Exception as e:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, f"Animation error: {str(e)}", True


# Add this callback to handle canvas drawing
app.clientside_callback(
    """
    function(animation_data_json) {
        if (!animation_data_json) {
            return window.dash_clientside.no_update;
        }
        
        try {
            const data = JSON.parse(animation_data_json);
            const canvas = document.getElementById('animation-canvas');
            if (!canvas) return window.dash_clientside.no_update;
            
            const ctx = canvas.getContext('2d');
            const people = data.people || [];
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Draw people
            people.forEach(person => {
                ctx.beginPath();
                ctx.arc(person.x, person.y, person.radius, 0, 2 * Math.PI);
                
                // Set color based on status
                if (person.status === 'susceptible') {
                    ctx.fillStyle = '#3498db';  // Blue
                } else if (person.status === 'infected') {
                    ctx.fillStyle = '#e74c3c';  // Red
                } else if (person.status === 'recovered') {
                    ctx.fillStyle = '#27ae60';  // Green
                }
                
                ctx.fill();
                
                // Add border
                ctx.strokeStyle = '#2c3e50';
                ctx.lineWidth = 1;
                ctx.stroke();
            });
            
            return window.dash_clientside.no_update;
        } catch (e) {
            console.error('Canvas drawing error:', e);
            return window.dash_clientside.no_update;
        }
    }
    """,
    Output('animation-canvas', 'id'),  # Dummy output
    Input('animation-data', 'children')
)

# Helper functions for creating figures and displays
def create_static_figure(time_points, S, I, R, metrics):
    """Create static figure with complete data"""
    fig = go.Figure()
    
    # Add all traces with complete data
    fig.add_trace(go.Scatter(
        x=time_points, y=S, mode='lines', name='Susceptible (S)',
        line=dict(color='#3498db', width=3),
        hovertemplate='Day %{x}<br>Susceptible: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points, y=I, mode='lines', name='Infected (I)',
        line=dict(color='#e74c3c', width=3),
        hovertemplate='Day %{x}<br>Infected: %{y:,.0f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points, y=R, mode='lines', name='Recovered (R)',
        line=dict(color='#27ae60', width=3),
        hovertemplate='Day %{x}<br>Recovered: %{y:,.0f}<extra></extra>'
    ))
    
    # Add peak infection marker
    fig.add_trace(go.Scatter(
        x=[metrics['peak_day']], y=[metrics['peak_infected']],
        mode='markers', name='Peak Infection',
        marker=dict(color='red', size=10, symbol='star'),
        hovertemplate='Peak Day: %{x}<br>Peak Infected: %{y:,.0f}<extra></extra>'
    ))
    
    # Update layout
    max_pop = max(S[0], np.max(I), np.max(R)) * 1.1
    fig.update_layout(
        title='SIR Model - Complete Simulation',
        xaxis={'title': 'Days', 'range': [0, len(time_points)]},
        yaxis={'title': 'Number of People', 'range': [0, max_pop]},
        hovermode='x unified',
        height=500,
        legend=dict(x=0.7, y=0.95),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_empty_figure(N, I0):
    """Create empty figure for initial state"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[0], y=[N - I0], mode='lines', name='Susceptible (S)',
        line=dict(color='#3498db', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0], y=[I0], mode='lines', name='Infected (I)',
        line=dict(color='#e74c3c', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=[0], y=[0], mode='lines', name='Recovered (R)',
        line=dict(color='#27ae60', width=3)
    ))
    
    fig.update_layout(
        title='SIR Model - Ready to Simulate',
        xaxis={'title': 'Days'},
        yaxis={'title': 'Number of People'},
        hovermode='x unified',
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_metrics_display(metrics):
    """Create metrics display components"""
    return [
        html.P(f"R₀ (Basic Reproduction Number): {metrics['R0']}", style={'margin': '5px 0'}),
        html.P(f"Peak Infections: {metrics['peak_infected']:,}", style={'margin': '5px 0'}),
        html.P(f"Peak Day: {metrics['peak_day']}", style={'margin': '5px 0'}),
        html.P(f"Final Recovered: {metrics['final_recovered']:,}", style={'margin': '5px 0'})
    ]

def create_empty_metrics():
    """Create empty metrics display"""
    return [
        html.P("R₀ (Basic Reproduction Number): --", style={'margin': '5px 0'}),
        html.P("Peak Infections: --", style={'margin': '5px 0'}),
        html.P("Peak Day: --", style={'margin': '5px 0'}),
        html.P("Final Recovered: --", style={'margin': '5px 0'})
    ]

# Add at the end of the file
if __name__ == '__main__':
    app.run(debug=True)