import base64
import json

import dash
import plotly.graph_objects as go
from cv2 import cv2
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from fer import FER

from Video.video import VideoCamera


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
            # dcc.Graph(figure=px.bar(), id='waveform'),
            dcc.Graph(
                figure={},
                id='predictions',
                animate=False),
        ], style={'display': 'flex', 'flex-direction': 'column'}),
    ], style={'display': 'flex', 'flex-direction': 'row', "justify-content": "center"}),
    dcc.Interval(
        id='video-interval',
        disabled=False,
        interval=50,  # in milliseconds
        n_intervals=0
    ),
    dcc.Interval(
        id='prediction-interval',
        disabled=False,
        interval=2000,  # in milliseconds
        n_intervals=0
    ),
    # dcc.Store stores the bounding box so that it can be shared to other callbacks
    dcc.Store(id='bounding-box')
], style={"height": "100%", "width": "100%", "justify-content": "center"})

camera = VideoCamera(0)  # Set to 0 to capture webcam feed
detector = FER()  # Detector and predictor


@app.callback(Output('bounding-box', 'data'),
              Input('video-interval', 'n_intervals'))
def retrieve_frame_data(n):
    frame = camera.next_frame()
    # Re-encode the frame as a JPEG then encode it into Base64
    # Allows the image data to be in a string format
    # This string can be passed directly into the HTML "src" attribute for the image
    # Pass this string to the "src" parameter for the vidfeed
    # https://community.plotly.com/t/does-dash-support-opencv-video-from-webcam/11012/11
    output = detector.find_faces(frame)
    bb = output[0].tolist() if len(output) > 0 else None
    return json.dumps(bb)


@app.callback(Output('vidfeed', 'src'),
              Input('bounding-box', 'data'))
def update_video_feed(data):
    bb = json.loads(data)
    frame = camera.current_frame()
    if bb is not None:
        frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[0] + bb[2], bb[1] + bb[3]),
                              color=(121, 49, 132), thickness=3)
    _, jpeg = cv2.imencode('.jpeg', cv2.flip(frame, 1))
    return f"data:image/jpeg;base64, {base64.b64encode(jpeg.tobytes()).decode()}"


@app.callback(Output('predictions', 'figure'),
              [Input('prediction-interval', 'n_intervals'),
               Input('bounding-box', 'data')])
def update_video_predictions(n, data):
    if n < 1:
        raise PreventUpdate()

    bb = json.loads(data)
    if bb is None:
        raise PreventUpdate()

    frame = camera.current_frame()
    predictions = detector.detect_emotions(frame, [bb])[0]["emotions"]
    return go.Figure(data={
        "data": [{"type": "bar",
                  "x": list(predictions.keys()),
                  "y": list(predictions.values())}],
        "layout": {"title": {"text": "Predictions"},
                   "template": "plotly_dark"}
    })


if __name__ == "__main__":
    app.run_server(debug=True)
