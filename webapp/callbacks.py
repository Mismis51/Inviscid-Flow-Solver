from dash import html
from dash import Input, Output, State, callback_context, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

import numpy as np
import base64
import io
import csv
from copy import deepcopy

from .utils import compute_flow
from src import geometry, linear_vortex_solver, compute_coefficients

# Constants
ANGLE_RANGE = (-15, 15)
NUM_ANGLES = 61
DEFAULT_VERTICES = 4

def register_callbacks(app, default_geometry, default_solver):
    #region Angle Synchronization Callback
    @app.callback(
        [Output('angle-slider', 'value'),
         Output('angle-input', 'value')],
        [Input('angle-slider', 'value'),
         Input('angle-input', 'value')],
        prevent_initial_call=True
    )
    def sync_angle_values(slider_value, input_value):
        """Synchronize slider and input field for angle of attack control."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        if input_value is None:
            input_value = 0

        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'angle-slider':
            return [no_update, round(slider_value, 3)]
        
        try:
            numeric_value = float(input_value)
        except ValueError:
            return [no_update, no_update]

        clamped_value = np.clip(numeric_value, *ANGLE_RANGE)
        return [clamped_value, input_value]
    #endregion

    #region File Handling Callbacks
    @app.callback(
        [Output('upload-status', 'children'),
         Output('geometry-store', 'data'),
         Output('upload-data', 'contents')],
        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
        prevent_initial_call=True
    )
    def handle_file_upload(file_content, filename):
        """Process uploaded airfoil data files."""
        if not file_content:
            raise PreventUpdate

        try:
            _, content_string = file_content.split(',')
            decoded_content = base64.b64decode(content_string).decode('utf-8')
            success_message = html.Div([
                html.Span("Successfully uploaded "),
                html.B(filename),
                html.Span(" for processing.")
            ])
            return success_message, decoded_content, None
        except Exception as e:
            error_message = html.Div([
                html.Span("File processing error: "),
                html.B(str(e))
            ])
            return error_message, None, None

    @app.callback(
        [Output('geometry-vertex', 'data', allow_duplicate=True),
         Output('solver-rhs', 'data', allow_duplicate=True)],
        [Input('geometry-store', 'data'),
         Input('nb-vertex-input', 'value')],
        prevent_initial_call=True
    )
    def update_airfoil_geometry(file_content, num_vertices):
        """Update stored geometry data from uploaded file."""
        num_vertices = num_vertices or DEFAULT_VERTICES
        airfoil = geometry(nb_vertex=num_vertices)
        airfoil.load_txt(io.StringIO(file_content))
        solver = linear_vortex_solver(airfoil)
        return airfoil.vertex.tolist(), solver.RHS.tolist()
    #endregion

    #region NACA Generation Callback
    @app.callback(
        [Output('geometry-vertex', 'data', allow_duplicate=True),
         Output('solver-rhs', 'data', allow_duplicate=True)],
        Input('generate-naca-button', 'n_clicks'),
        [State('naca-m', 'value'),
         State('naca-p', 'value'),
         State('naca-t', 'value'),
         State('nb-vertex-input', 'value')],
        prevent_initial_call=True
    )
    def generate_naca_profile(click_count, m, p, t, num_vertices):
        """Generate NACA 4-digit airfoil profile from user inputs."""
        if not click_count:
            raise PreventUpdate

        try:
            m = m or 0
            p = p or 1
            t = t or 0
            num_vertices = num_vertices or DEFAULT_VERTICES

            airfoil = geometry(nb_vertex=num_vertices)
            airfoil.load_naca(m/100, p/10, t/100)
            
            solver = linear_vortex_solver(airfoil)
            return airfoil.vertex.tolist(), solver.RHS.tolist()
            
        except Exception as e:
            return [no_update, f"Generation error: {str(e)}"]
    #endregion

    #region Aerodynamic Visualization Callbacks
    def _create_aero_figure(x_data, y_data, title, y_label):
        """Helper to create standardized coefficient plots."""
        return go.Figure(
            data=[go.Scatter(x=x_data, y=y_data, 
                        mode='lines+markers', name=y_label)],
            layout=go.Layout(
                title=title,
                xaxis_title='Angle of Attack (°)',
                yaxis_title=y_label,
                height=400
            )
        )

    @app.callback(
        [Output('cl-graph', 'figure'),
        Output('cd-graph', 'figure'),
        Output('cm-graph', 'figure'),
        Output('precomputed-aero-data', 'data')],
        Input('geometry-vertex', 'data'),
        [State('solver-rhs', 'data'),
        State('nb-vertex-input', 'value')]
    )
    def update_aerodynamic_coefficients(vertices, solver_data, num_vertices):
        """Update precomputed aerodynamic coefficient plots."""
        # Handle default values
        num_vertices = num_vertices or DEFAULT_VERTICES
        
        # Create geometry instance based on current state
        if vertices:
            current_geometry = geometry(nb_vertex=num_vertices)
            current_geometry.vertex = np.array(vertices)
            current_geometry._compute_parameters()
        else:
            current_geometry = deepcopy(default_geometry)
        
        # Configure solver with current geometry
        current_solver = deepcopy(default_solver)
        current_solver.geometry = current_geometry
        if solver_data is not None:  # Preserve solver RHS if available
            current_solver.RHS = np.array(solver_data)
        current_solver._create_normals()  # Critical normalization step
        
        # Compute coefficients across angle range
        angles = np.linspace(*ANGLE_RANGE, NUM_ANGLES)
        cl_values, cd_values, cm_values = [], [], []
        
        for angle in angles:
            current_geometry.set_angle_deg(-angle)
            vortex_strengths = current_solver.solve(-current_geometry.angle)
            cl, cd, cm = compute_coefficients(current_geometry, vortex_strengths)
            cl_values.append(cl)
            cd_values.append(cd)
            cm_values.append(cm)
        
        # Package data for storage and return
        aero_data = {
            'angles': angles.tolist(),
            'Cl': cl_values,
            'Cd': cd_values,
            'Cm': cm_values
        }

        return (
            _create_aero_figure(angles, cl_values, 'Lift Coefficient', 'Cl'),
            _create_aero_figure(angles, cd_values, 'Drag Coefficient', 'Cd'),
            _create_aero_figure(angles, cm_values, 'Moment Coefficient', 'Cm'),
            aero_data
        )

    @app.callback(
        [Output('flow-graph', 'figure'),
        Output('cp-graph', 'figure'),
        Output('cl-value', 'children'),
        Output('cd-value', 'children'),
        Output('cm-value', 'children'),
        Output('current-cp-data', 'data')],
        [Input('angle-input', 'value'),
        Input('plot-type', 'value'),
        Input('geometry-vertex', 'data'),
        Input('resolution-input', 'value')],
        [State('solver-rhs', 'data'),
        State('nb-vertex-input', 'value')]
    )
    def update_flow_visualization(angle, plot_mode, vertices, 
                                resolution, solver_data, num_vertices):
        """Main visualization update for flow fields and coefficients."""
        # Handle input validation and defaults
        try:
            numeric_angle = float(angle)
        except (ValueError, TypeError):
            numeric_angle = 0.0
        
        num_vertices = num_vertices or DEFAULT_VERTICES
        resolution = resolution or 200  # Default resolution if not provided
        
        # Create geometry instance based on current state
        if vertices:
            current_geometry = geometry(nb_vertex=num_vertices)
            current_geometry.vertex = np.array(vertices)
            current_geometry._compute_parameters()
        else:
            current_geometry = deepcopy(default_geometry)
        
        # Configure solver with current geometry
        current_solver = deepcopy(default_solver)
        current_solver.geometry = current_geometry
        if solver_data is not None:
            current_solver.RHS = np.array(solver_data)
        current_solver._create_normals()  # Essential for accurate solutions
        
        # Compute aerodynamic properties
        current_geometry.set_angle_deg(-numeric_angle)
        vortex_strengths = current_solver.solve(-current_geometry.angle)
        airfoil_points = current_geometry.get_rotated_vertex()
        cp_values = 1 - np.square(vortex_strengths)
        
        # Calculate coefficients and flow data
        cl, cd, cm = compute_coefficients(current_geometry, vortex_strengths)
        x_grid, y_grid, flow_data = compute_flow(
            plot_mode, vortex_strengths, airfoil_points, resolution
        )
        
        # Generate visualization figures
        flow_figure = _create_flow_plot(x_grid, y_grid, flow_data, 
                                    plot_mode, numeric_angle, airfoil_points)
        cp_figure = _create_cp_plot(airfoil_points, cp_values, numeric_angle)
        
        return (
            flow_figure,
            cp_figure,
            f"{cl:.4f}", f"{cd:.4f}", f"{cm:.4f}",
            {'x_pos': airfoil_points[1:-1, 0].tolist(),
            'Cp': cp_values[1:-1].tolist()}
        )
    #endregion

    def _create_flow_plot(x_grid, y_grid, data, mode, angle, airfoil):
        """Create flow visualization plot (streamlines/velocities)."""
        # Create the appropriate trace type based on visualization mode
        if mode == 'streamlines':
            flow_trace = go.Contour(
                x=x_grid[0], 
                y=y_grid[:,0], 
                z=data,
                zmin=-1, 
                zmax=1, 
                colorscale='RdBu',
                contours=dict(showlines=True, showlabels=False),
                colorbar=dict(title="Streamfunction", x=1.05)
            )
        else:  # velocities
            flow_trace = go.Heatmap(
                x=x_grid[0], 
                y=y_grid[:,0], 
                z=data,
                zmin=0, 
                zmax=2, 
                colorscale='Viridis',
                colorbar=dict(title="Velocity Magnitude", x=1.05)
            )

        # Airfoil overlay
        airfoil_trace = go.Scatter(
            x=airfoil[:,0], 
            y=airfoil[:,1],
            mode='lines', 
            line=dict(color='black', width=2),
            fill='toself', 
            fillcolor='rgba(0,0,0,0.2)'
        )

        return go.Figure(data=[flow_trace, airfoil_trace]).update_layout(
            title=f"{mode.capitalize()} at {angle:.3f}°",
            xaxis_title="x-axis", 
            yaxis_title="y-axis",
            width=700, 
            height=600,
            xaxis_range=[-0.5, 1.5], 
            yaxis_range=[-1, 1]
        )

    def _create_cp_plot(airfoil, cp_values, angle):
        """Create pressure coefficient distribution plot."""
        return go.Figure(
            data=[go.Scatter(
                x=airfoil[1:-1,0], y=cp_values[1:-1],
                mode='lines+markers',
                line=dict(color='blue', width=2),
                marker=dict(size=8, color='red')
            )]
        ).update_layout(
            title=f"Cp Distribution at {angle:.3f}°",
            xaxis_title="Chord Position (x)",
            yaxis_title="Cp Value",
            width=500, height=600,
            yaxis=dict(autorange="reversed")
        )
    #endregion

    #region Data Export Callback
    @app.callback(
        Output("download-data", "data"),
        Input("btn-download", "n_clicks"),
        [State('current-cp-data', 'data'),
         State('precomputed-aero-data', 'data'),
         State('angle-input', 'value')],
        prevent_initial_call=True
    )
    def export_aerodynamic_data(click_count, cp_data, aero_data, current_angle):
        """Export current and precomputed data to CSV."""
        if not click_count:
            raise PreventUpdate

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        
        # Current pressure data
        writer.writerow([f"CP Distribution at {current_angle}°"])
        writer.writerow(["X Position", "CP Value"])
        writer.writerows(zip(cp_data['x_pos'], cp_data['Cp']))
        
        # Precomputed aero data
        writer.writerow(["\nPrecomputed Aerodynamic Coefficients"])
        writer.writerow(["Angle (°)", "Cl", "Cd", "Cm"])
        writer.writerows(zip(aero_data['angles'], 
                           aero_data['Cl'], 
                           aero_data['Cd'], 
                           aero_data['Cm']))

        return {
            'content': buffer.getvalue(),
            'filename': "aerodynamic_analysis.csv"
        }
    #endregion
