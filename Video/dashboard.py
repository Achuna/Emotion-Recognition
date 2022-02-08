import base64

import dash
import plotly.express as px
import plotly.graph_objects as go
from cv2 import cv2
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from fer import FER
from plotly.subplots import make_subplots

from Video.video import VideoCamera


# def build_dashboard():
#     fig = make_subplots(
#         rows=4, cols=5, specs=[
#             [{"type": "image", "rowspan": 4, "colspan": 3}, None, None,
#              {"type": "bar", "rowspan": 2, "colspan": 2}, None],
#             [None, None, None, None, None],
#             [None, None, None, {"type": "bar", "rowspan": 2, "colspan": 2}, None],
#             [None, None, None, None, None]
#         ]
#     )
#     fig.add_trace(go.Image(name="vidfeed"), row=1, col=1)
#     fig.add_trace(go.Bar(name="visualizer"), row=1, col=4)
#     fig.add_trace(go.Bar(name="predictions"), row=3, col=4)
#     fig.update_layout(template="plotly_dark")
#
#     return fig


def main():
    app = dash.Dash(__name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
                    update_title=None)
    app.layout = html.Div([
        html.H1(children='Emotion Recognition!', style={'textAlign': 'center'}),

        html.Div(children='Dashboard under active development; subject to change.', style={
            'textAlign': 'center',
        }),

        html.Div([
            html.Div([
                html.Img(id="vidfeed", style={"max-width": "100%", "max-height": "100%"}),
            ], style={"aspect-ratio": "1/1", "padding": "5%"}),
            html.Div([
                dcc.Graph(figure=px.bar(), id='waveform'),
                dcc.Graph(figure=px.bar(), id='predictions')
            ], style={'display': 'flex', 'flex-direction': 'column'}),
        ], style={'display': 'flex', 'flex-direction': 'row', "justify-content": "center"}),
        dcc.Interval(
            id='interval-component',
            interval=50,  # in milliseconds
            n_intervals=0
        )
    ], style={"height": "100%", "width": "100%", "justify-content": "center"})

    camera = VideoCamera(0)  # Set to 0 to capture webcam feed

    @app.callback(Output('vidfeed', 'src'),
                  Input('interval-component', 'n_intervals'))
    def update_video_feed(n):
        frame = camera.capture_frame()
        # Re-encode the frame as a JPEG then encode it into Base64
        # Allows the image data to be in a string format
        # This string can be passed directly into the HTML "src" attribute for the image
        # Pass this string to the "src" parameter for the vidfeed
        # https://community.plotly.com/t/does-dash-support-opencv-video-from-webcam/11012/11
        _, jpeg = cv2.imencode('.jpeg', cv2.flip(frame, 1))
        return f"data:image/jpeg;base64, {base64.b64encode(jpeg.tobytes()).decode()}"

    app.run_server(debug=True)


main()
