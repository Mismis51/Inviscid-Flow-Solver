import dash
import numpy as np
import plotly.graph_objs as go
from src import geometry, linear_vortex_solver, compute_coefficients
import webapp

app = dash.Dash(__name__)

# Initialize default geometry and solver
default_geometry = geometry(nb_vertex=256)
default_geometry.load_txt('examples/0012.dat')
default_solver = linear_vortex_solver(default_geometry)

# Precompute aerodynamic coefficients
precomputed_angles = np.linspace(-15, 15, 61)
cl_values, cd_values, cm_values = [], [], []

for alpha_deg in precomputed_angles:
    default_geometry.set_angle_deg(-alpha_deg)
    gammas = default_solver.solve(-default_geometry.angle)
    Cl, Cd, Cm = compute_coefficients(default_geometry, gammas)
    cl_values.append(Cl)
    cd_values.append(Cd)
    cm_values.append(Cm)

# Create coefficient figures
def create_coefficient_figure(x_data, y_data, title, y_label):
    return go.Figure(
        data=[go.Scatter(x=x_data, y=y_data, mode='lines+markers', name=y_label)],
        layout={
            'title': title,
            'xaxis_title': 'Angle of Attack (°)',
            'yaxis_title': y_label,
            'height': 400
        }
    )

cl_figure = create_coefficient_figure(
    precomputed_angles, cl_values, 'Lift Coefficient vs Angle of Attack', 'Cl'
)
cd_figure = create_coefficient_figure(
    precomputed_angles, cd_values, 'Drag Coefficient vs Angle of Attack', 'Cd'
)
cm_figure = create_coefficient_figure(
    precomputed_angles, cm_values, 'Moment Coefficient vs Angle of Attack', 'Cm'
)

webapp.app_layout(app, cl_figure, cd_figure, cm_figure)
webapp.register_callbacks(app, default_geometry, default_solver)

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")
