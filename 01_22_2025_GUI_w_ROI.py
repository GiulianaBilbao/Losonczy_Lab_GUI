import dash
import pickle
import os
from dash import html, dcc
import plotly.graph_objs as go
import numpy as np
import json
import base64  # For decoding uploaded file contents
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__)

# Placeholder for ROI-separated events
roi_events = {}
roi_traces = {}

#CHANGE THE NAME OF THE FILE LOCATION HERE!!!!
#_____________________________________________
#_____________________________________________
uploaded_file_path = r'C:\Users\giuliana\Desktop\Voltage_Labeling_GUI\formatted_data_for_GUI_with_ROI'

app.layout = html.Div([
    html.H1("Event Curation Tool"),

    # Add and Remove Event Buttons with Dropdown
    html.Div([
        html.Div([
            html.Button("Add Event", id="add-event-btn", 
                        style={'marginBottom': '10px', 
                               'width': '150px',
                               'height': '40px'
                               }),
            dcc.Dropdown(
                id="event-type-dropdown",
                options=[
                    {'label': 'Fast Only', 'value': 'fast_only'},
                    {'label': 'Slow 1AP', 'value': 'slow_1AP'},
                    {'label': 'Burst', 'value': 'burst'},
                    {'label': 'Slow Only', 'value': 'slow_only'},
                ],
                placeholder="Select Event Type",
                style={'width': '150px'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '20px'}),
        html.Button("Remove Event", 
                    id="remove-event-btn", 
                    style={'width': '150px',
                           'height': '40px',}),
    ], style={'float': 'right', 'marginBottom': '20px'}),

    # File upload component
    dcc.Upload(
        id='upload-data',
        children=html.Button("Upload Data File"),
        style={
            'display': 'inline-block',
            'marginBottom': '10px',
            'height': '40px'
        },
        multiple=False  # Allow only one file
    ),

    html.Div([
        html.Button("Load Dataset", id="load-dataset-btn", style={
                'height': '40px',
                'marginTop': '10px',
                'marginRight': '20px'  # Add space between buttons
        }),
        html.Button("Reload Visualization", id="reload-visualization-btn", style={
                'height': '40px',
                'marginTop': '10px',  # Match the alignment
                'marginBottom': '20px',
                'width': '200px',
                'marginRight': '20px'
        }),

        # Export data button
        html.Button("Export Data", 
            id="export-data-btn", 
            style={
                'marginBottom': '20px',
                'height': '40px',
                }),

    ], style={'marginBottom': '20px'}),


    # Placeholder for status messages
    html.Div(id='status-message', style={
        'color': 'green',
        'marginTop': '10px',
        'textAlign': 'center'
    }),

    # Container for dynamically generated graphs
    html.Div(id='roi-graphs-container', style={'marginTop': '20px'}),

    # Store for current selection
    dcc.Store(id='current-selection-store', data={})

])


@app.callback(
    [Output('status-message', 'children'),
     Output('roi-graphs-container', 'children'),
     Output('current-selection-store', 'data')],
    [Input('load-dataset-btn', 'n_clicks'),
     Input({'type': 'roi-graph', 'index': dash.dependencies.ALL}, 'selectedData')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename')],  # Get the uploaded file's name
    prevent_initial_call=True
)
def handle_load_and_interactions(n_clicks, selected_data_list, contents, filename):
    """
    Handle dataset loading and graph interactions, updating current selection.
    """
    global roi_events, roi_traces

    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    print(f"ctx.triggered: {ctx.triggered}")

    # Identify the trigger
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger == 'load-dataset-btn':
        if not contents or not filename:
            return "No dataset uploaded. Please upload a dataset first.", [], {}

        # Pass both contents and filename to the function
        message, graphs = load_and_generate_graphs(contents, filename)
        return message, graphs, {}

    if 'type' in trigger:
        message, current_selection = handle_roi_graph_interaction(trigger, ctx, roi_events)
        print(f"Current Selection Updated: {current_selection}")
        return message, dash.no_update, current_selection

    return dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output('status-message', 'children', allow_duplicate=True),
    Input('remove-event-btn', 'n_clicks'),
    State('current-selection-store', 'data'),
    prevent_initial_call=True
)
def handle_remove_event(n_clicks, current_selection):
    """
    Handle removing events based on current selection stored in dcc.Store.
    """
    if not current_selection:
        return "No selection available to remove events. Please select a range first."

    message = remove_event(current_selection)
    return message


@app.callback(
    Output('status-message', 'children', allow_duplicate=True),
    Input('add-event-btn', 'n_clicks'),
    State('event-type-dropdown', 'value'),
    State('current-selection-store', 'data'),
    prevent_initial_call=True
)
def handle_add_event(n_clicks, selected_event_type, current_selection):
    global roi_events

    print(f"Current Selection in Add Event: {current_selection}")
    print(f"Selected Event Type: {selected_event_type}")

    # Validate selection and event type
    if not selected_event_type:
        return "Please select an event type from the dropdown before adding an event."
    if not current_selection or 'roi' not in current_selection or 'events' not in current_selection:
        return "No valid selection available to add events. Please box select a range on a graph."

    roi = current_selection['roi']
    if roi not in roi_events:
        return f"ROI {roi} not found in dataset."

    box_selected_events = current_selection['events']
    if not box_selected_events:
        return "No valid range selected for adding events. Please box select an area."

    # Ensure event type exists and is properly initialized
    if selected_event_type not in roi_events[roi]:
        roi_events[roi][selected_event_type] = np.empty((0, 2), dtype=float)  # Correctly initialize as a 2D array
        print(f"Initialized roi_events[{roi}][{selected_event_type}] with shape {roi_events[roi][selected_event_type].shape}")

    # Ensure it remains a 2D array
    if roi_events[roi][selected_event_type].ndim != 2 or roi_events[roi][selected_event_type].shape[1] != 2:
        print(f"Fixing shape of roi_events[{roi}][{selected_event_type}] to be a 2D array.")
        roi_events[roi][selected_event_type] = roi_events[roi][selected_event_type].reshape(0, 2)

    # Add a new event with the correct type
    box_event = box_selected_events[0]  # Directly access the first event
    print(f"Processing Box Event Before Update: {box_event}")

    # Explicitly set the event type in current_selection
    current_selection['events'][0]['type'] = selected_event_type  # Update the type directly
    print(f"Processing Box Event After Update: {current_selection['events'][0]}")

    # Ensure the new event is a 2D array
    new_event = np.array(box_event['event']).reshape(1, -1)  # Reshape to ensure it's 2D
    print(f"New event reshaped: {new_event} with shape {new_event.shape}")

    # Append the new event to the corresponding event type
    try:
        print(f"roi_events[{roi}][{selected_event_type}] before append: {roi_events[roi][selected_event_type]}")
        print(f"Shape of roi_events[{roi}][{selected_event_type}]: {roi_events[roi][selected_event_type].shape}")
        print(f"New event to append: {new_event}")
        print(f"Shape of new_event: {new_event.shape}")

        roi_events[roi][selected_event_type] = np.append(
            roi_events[roi][selected_event_type], new_event, axis=0
    )
        print(f"Added event to {selected_event_type}: {new_event}")
    except Exception as e:
        print(f"Error while appending event: {e}")
        return f"Failed to add event to ROI {roi}. Error: {str(e)}"
    
    # Debug updated roi_events
    print(f"Updated ROI Events for {roi}: {roi_events[roi]}")

    return f"Event of type '{selected_event_type}' added to ROI {roi} for range {box_event['event']}."


##RELOADING THE SPECIFIC ROI PLOT AFTER EVENTS HAVE BEEN EDITED

@app.callback(
    Output({'type': 'roi-graph', 'index': dash.dependencies.ALL}, 'figure', allow_duplicate=True),
    Input('reload-visualization-btn', 'n_clicks'),
    State('current-selection-store', 'data'),
    State({'type': 'roi-graph', 'index': dash.dependencies.ALL}, 'figure'),
    prevent_initial_call=True
)
def reload_roi_graph(n_clicks, current_selection, all_figures):
    """
    Reload the graph for the specific ROI based on the updated events and preserve x-axis range.
    """
    global roi_events, roi_traces

    # Validate current selection
    if not current_selection or 'roi' not in current_selection:
        return [dash.no_update] * len(all_figures)  # No valid selection, no updates

    selected_roi = current_selection['roi']

    if selected_roi not in roi_events or selected_roi not in roi_traces:
        return [dash.no_update] * len(all_figures)  # ROI not found in current data

    updated_figures = []
    for fig in all_figures:
        # Check if the figure corresponds to the selected ROI
        roi = fig['layout']['title']['text'].split(":")[1].strip()  # Extract ROI from the title
        if roi == selected_roi:
            # Fetch current x-axis range
            current_x_range = fig['layout']['xaxis'].get('range', [0, 5])  # Default range if not set

            # Rebuild the figure
            amplitude = roi_traces[roi]['amplitude']
            time = roi_traces[roi]['time']

            updated_fig = go.Figure()

            # Add amplitude trace
            updated_fig.add_trace(go.Scatter(
                x=time,
                y=amplitude,
                mode='lines',
                name='Amplitude'
            ))

            # Add updated event markers for each event type
            for event_type, events in roi_events[roi].items():
                color = get_event_color(event_type)  # Use the helper function
                add_event_markers(updated_fig, events, event_type, color)

            updated_fig.update_layout(
                title=f"ROI: {roi}",
                xaxis_title="Time (s)",
                yaxis_title="Amplitude",
                height=300,
                margin={'l': 40, 'r': 10, 't': 30, 'b': 40},
                xaxis=dict(range=current_x_range),  # Apply the current x-axis range
            )

            updated_figures.append(updated_fig)
        else:
            # For non-selected ROIs, leave the figure unchanged
            updated_figures.append(dash.no_update)

    return updated_figures


#EXPORTING THE DATA ________________________________________________________________________
#___________________________________________________________________________________________
@app.callback(
    Output('status-message', 'children', allow_duplicate=True),
    Input('export-data-btn', 'n_clicks'),
    prevent_initial_call=True
)
def export_data(n_clicks):
    """
    Export the ROI events and traces data to a .pkl file with the correct structure.
    """
    global roi_events, roi_traces, uploaded_file_path

    if not roi_events or not roi_traces:
        return "No data available to export. Please upload and process a file first."

    # Determine the export path
    if uploaded_file_path:
        base_name = os.path.splitext(uploaded_file_path)[0]
        export_path = f"{base_name}_new.pkl"
    else:
        export_path = os.path.join(os.getcwd(), "exported_data.pkl")

    # Save the data using pickle
    try:
        # Ensure the exported data matches the input format
        export_data = {
            "annotations": {
                roi: {
                    "trace": {
                        "amplitude": roi_traces[roi]['amplitude'].tolist(),
                        "time": roi_traces[roi]['time'].tolist()
                    },
                    "fast_only": {"locations": roi_events[roi]['fast_only'].tolist()},
                    "slow_1AP": {"locations": roi_events[roi]['slow_1AP'].tolist()},
                    "burst": {"locations": roi_events[roi]['burst'].tolist()},
                    "slow_only": {"locations": roi_events[roi]['slow_only'].tolist()}
                }
                for roi in roi_events
            }
        }

        # Debug: Verify the structure before saving
        print("Exporting Data Structure:", export_data)

        with open(export_path, 'wb') as f:
            pickle.dump(export_data, f)

        return f"Data exported successfully to '{export_path}'."
    except Exception as e:
        return f"Error exporting data: {str(e)}"

#LEFT AND RIGHT ARROW BUTTON ____________________________________________________________________
#________________________________________________________________________________________________
@app.callback(
    Output({'type': 'roi-graph', 'index': dash.dependencies.MATCH}, 'figure'),
    [Input({'type': 'left-btn', 'index': dash.dependencies.MATCH}, 'n_clicks'),
     Input({'type': 'right-btn', 'index': dash.dependencies.MATCH}, 'n_clicks')],
    [State({'type': 'roi-graph', 'index': dash.dependencies.MATCH}, 'figure')],
    prevent_initial_call=True
)
def update_plot_range(left_clicks, right_clicks, figure):
    """
    Update the x-axis range of the plot for the specific ROI based on button clicks.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # Identify the triggered button
    triggered_id = ctx.triggered[0]['prop_id']
    direction = 'left' if 'left-btn' in triggered_id else 'right'

    print(f"Triggered ID: {triggered_id}")
    print(f"Direction: {direction}")
    print(f"Figure before update: {figure}")

    # Update the x-axis range
    x_range = figure['layout']['xaxis']['range'] or [0, 5]  # Default range if not set
    shift = 0.3 if direction == 'right' else -0.3
    new_range = [x_range[0] + shift, x_range[1] + shift]
    figure['layout']['xaxis']['range'] = new_range

    print(f"New x-axis range: {new_range}")
    return figure
#FUNCTIONS_____________________________________________________________________________________
#______________________________________________________________________________________________

def remove_event(current_selection):
    """
    Remove selected events from the specified ROI.

    Parameters:
    - current_selection (dict): Contains ROI and events selected for removal.

    Returns:
    - str: Status message indicating the result of the action.
    """
    global roi_events

    # Ensure current_selection is provided
    if not current_selection or 'roi' not in current_selection or 'events' not in current_selection:
        return "No valid selection available to remove events. Please select a range first."

    roi = current_selection['roi']
    events_to_remove = current_selection['events']

    # Debugging
    print(f"Removing events from ROI: {roi}")
    print(f"Events to remove: {events_to_remove}")

    # Ensure ROI exists in roi_events
    if roi not in roi_events:
        return f"ROI {roi} not found in dataset."

    removed_any = False

    # Iterate through event types and remove matching events
    for event in events_to_remove:
        event_type = event['type']
        event_data = event['event']

        # Ensure event type exists in the ROI
        if event_type in roi_events[roi]:
            # Remove events matching the range
            event_list = roi_events[roi][event_type]
            mask = ~np.all(event_list == event_data, axis=1)  # Filter out the matching event
            filtered_events = event_list[mask]

            # Check if any events were removed
            if len(filtered_events) < len(event_list):
                roi_events[roi][event_type] = filtered_events
                removed_any = True

    if removed_any:
        return f"Events successfully removed from ROI {roi}."
    return f"No matching events found to remove in ROI {roi}."



def add_event(event_type, selected_data_list, ctx):
    """
    Add a new event of the specified type to the selected ROI and range.

    Parameters:
    - event_type: The type of event to add (e.g., 'fast_only', 'burst').
    - selected_data_list: List of selected data from all ROI graphs.
    - ctx: Dash callback context to identify the triggered graph.

    Returns:
    - str: Status message indicating the result of the action.
    """
    global roi_events

    if not event_type:
        return "Please select an event type before adding an event."
    if not selected_data_list or not any(selected_data_list):
        return "Please select a range on a graph to add an event."

    # Find the selected graph and range
    for idx, selected_data in enumerate(selected_data_list):
        if selected_data and 'range' in selected_data:
            roi_index = ctx.inputs_list[1][idx]['id']['index']
            x_range = selected_data['range']['x']
            x0, x1 = x_range[0], x_range[1]

            # Add the new event
            if roi_index in roi_events and event_type in roi_events[roi_index]:
                roi_events[roi_index][event_type] = np.append(
                    roi_events[roi_index][event_type],
                    [[x0, x1]],
                    axis=0
                )
                return f"Event of type '{event_type}' added in ROI {roi_index} for range {x0:.2f} to {x1:.2f}."
    return "No valid selection found to add an event."



def handle_roi_graph_interaction(trigger, ctx, roi_events): 
    """
    Handle interactions with dynamically created ROI graphs.

    Parameters:
    - trigger: The triggered property ID from the callback context.
    - ctx: Dash callback context object.
    - roi_events: Dictionary containing event data for all ROIs.

    Returns:
    - str: Status message indicating the result of the interaction.
    - dict: Current selection with ROI and range (if any).
    """
    try:
        trigger_id = json.loads(trigger)
        print(f"Trigger ID: {trigger_id}")
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        return "Invalid trigger format.", {}

    if 'index' in trigger_id:
        selected_roi = trigger_id['index']

        # Extract the selected data for the triggered graph
        triggered_data = ctx.triggered[0]['value']  # Get the selectedData directly from ctx.triggered
        if triggered_data and 'range' in triggered_data:
            x_range = triggered_data['range']['x']
            x0, x1 = x_range[0], x_range[1]

            # Check for events in this range
            events = roi_events.get(selected_roi, {})  # Use .get() to avoid KeyError
            events_in_range = []

            # Collect event details including types
            for event_type, event_list in events.items():
                for event in event_list:
                    if x0 <= event[0] <= x1:
                        events_in_range.append({'type': event_type, 'event': event})

            if events_in_range:
                # Format the detected events for user feedback
                event_details = ', '.join([f"{e['type']} at {e['event']}" for e in events_in_range])
                return (
                    f"Box selected in ROI graph: {selected_roi}. "
                    f"Events detected in range {x0:.2f} to {x1:.2f}: {len(events_in_range)} "
                    f"({event_details})",
                    {'roi': selected_roi, 'events': events_in_range}
                )

            # No events in the range but valid selection
            current_selection = {
                'roi': selected_roi,
                'events': [{'type': None, 'event': [x0, x1]}]
            }
            return f"Box selected in ROI graph: {selected_roi}. Range: {x0:.2f} to {x1:.2f}.", current_selection

        # No valid box selection detected
        return f"User interacted with ROI graph: {selected_roi}. No valid box selection detected.", {}

    # No valid 'index' in the trigger
    return "Interaction detected, but no valid graph ID found.", {}



def detect_graph_interaction(selected_data_list):
    """
    Callback to detect interactions with ROI graphs.
    """
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
    if 'index' in trigger_id:
        return f"User interacted with ROI graph: {trigger_id['index']}"
    else:
        return "Interaction detected, but no valid graph ID found."


import pickle

def load_and_generate_graphs(contents, filename=None):
    global roi_events, roi_traces

    try:
        # Decode the uploaded dataset
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        # Determine file type based on filename
        if filename and filename.endswith('.pkl'):
            data = pickle.loads(decoded)
            #print("Loaded Data from .pkl File:", data)
        else:
            decoded = decoded.decode('utf-8')
            data = json.loads(decoded)

        if "annotations" not in data:
            print("Error: 'annotations' key not found in data.")
            return "Uploaded file does not contain valid annotations.", []

        annotations = data['annotations']
        roi_events = {}
        roi_traces = {}
        roi_graphs = []

        # Process each ROI
        for roi, details in annotations.items():
            roi_traces[roi] = {
                'amplitude': np.array(details['trace']['amplitude']),
                'time': np.array(details['trace']['time']),
            }

            roi_events[roi] = {
                'fast_only': np.array(details['fast_only']['locations']),
                'slow_1AP': np.array(details['slow_1AP']['locations']),
                'burst': np.array(details['burst']['locations']),
                'slow_only': np.array(details['slow_only']['locations']),
            }

            # Use the helper function to generate the plot
            fig = create_roi_plot(roi, details)

            roi_graphs.append(html.Div([
            dcc.Graph(
                id={'type': 'roi-graph', 'index': roi},
                figure=fig,
                style={'height': '300px', 'marginBottom': '10px'}
            ),
            html.Div([
                html.Button(
                    "Left",
                    id={'type': 'left-btn', 'index': roi},
                    style={
                        'marginRight': '20px',
                        'height': '40px',
                    }
                ),
                html.Button(
                    "Right",
                    id={'type': 'right-btn', 'index': roi},
                    style={
                        'marginRight': '20px',
                        'height': '40px',
                    }
                ),
            ], style={'textAlign': 'center', 'marginBottom': '20px'})
        ]))

        return f"Dataset successfully loaded. {len(roi_events)} ROIs processed.", roi_graphs

    except Exception as e:
        return f"Error loading dataset: {str(e)}", []


def create_roi_plot(roi, details, x_start=0, x_end=5):
    """
    Create a plotly graph for a specific ROI with its trace, event markers, and peaks for fast_only events.

    Parameters:
    - roi (str): The name of the ROI.
    - details (dict): The trace and event data for the ROI.

    Returns:
    - go.Figure: The generated plotly figure for the ROI.
    """
    trace = details['trace']
    amplitude = np.array(trace['amplitude'])
    time = np.array(trace['time'])

    events = {
        'fast_only': np.array(details['fast_only']['locations']),
        'slow_1AP': np.array(details['slow_1AP']['locations']),
        'burst': np.array(details['burst']['locations']),
        'slow_only': np.array(details['slow_only']['locations']),
    }

    fig = go.Figure()

    # Add amplitude trace
    fig.add_trace(go.Scatter(
        x=time,
        y=amplitude,
        mode='lines',
        name='Amplitude'
    ))

    # Add event markers and rectangles for each event type
    for event_type, color in zip(
        ['fast_only', 'slow_1AP', 'burst', 'slow_only'],
        ['red', 'green', 'blue', 'purple']
    ):
        if events[event_type].size > 0:  # Only add markers if there are events
            # Add markers at the midpoint of each event
            fig.add_trace(go.Scatter(
                x=[(onset + offset) / 2 for onset, offset in events[event_type]],
                y=[amplitude.max()] * len(events[event_type]),
                mode='markers',
                marker=dict(color=color, size=5),
                name=event_type.replace('_', ' ').title()  # Format event type name
            ))

            # Add transparent rectangles for the duration of the events
            for onset, offset in events[event_type]:
                fig.add_shape(
                    type="rect",
                    x0=onset,
                    x1=offset,
                    y0=amplitude.min(),
                    y1=amplitude.max(),
                    fillcolor=color,
                    opacity=0.4,  # Transparent color
                    line=dict(width=0)  # No border for the rectangle
                )

    # Add peaks for fast_only events using plot_peaks_roi
    if 'fast_only' in events and events['fast_only'].size > 0:
        plot_peaks_roi(
            fig,
            events['fast_only'],
            roi_amplitude=amplitude,
            roi_time=time,
            name='Fast Only',
            color='red'
        )

    # Update layout to ensure the legend is visible
    fig.update_layout(
        title=f"ROI: {roi}",
        xaxis=dict(range=[x_start, x_end]),
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
        margin={'l': 40, 'r': 10, 't': 30, 'b': 40},
        legend=dict(
            title="Event Types",
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
        )
    )

    return fig
    

def get_event_color(event_type):
    """
    Map event types to specific colors for visualization.

    Parameters:
    - event_type: The type of event (e.g., 'fast_only', 'slow_1AP').

    Returns:
    - str: The color associated with the event type.
    """
    color_map = {
        'fast_only': 'red',
        'slow_1AP': 'green',
        'burst': 'blue',
        'slow_only': 'purple'
    }
    return color_map.get(event_type, 'black')  # Default to black if event type is not recognized

def process_roi_box_select(selected_data_list, ctx, roi_events):
    if not ctx.triggered:
        return "No box select interaction detected."

    # Identify which graph triggered the callback
    trigger_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
    if 'index' not in trigger_id:
        return "Invalid trigger ID."

    selected_roi = trigger_id['index']
    selected_data = selected_data_list[trigger_id['index']]

    if not selected_data:
        return f"No selection made in ROI {selected_roi}."

    x_range = selected_data['range']['x']
    x0, x1 = x_range[0], x_range[1]

    events = roi_events[selected_roi]
    message, _ = process_box_select(selected_data, selected_roi, events)

    return f"Box selected in ROI {selected_roi} from {x0:.2f} to {x1:.2f}. {message}"


def add_event_markers(fig, events, name, color):
    """
    Adds event markers to a Plotly graph.

    Parameters:
    - fig: Plotly figure to modify.
    - events: Array of event onset-offset pairs.
    - name: Name of the event type (e.g., 'Fast Only').
    - color: Color for the event markers.
    """
    for event in events:
        if len(event) == 2:  # Ensure valid event format (onset, offset)
            onset, offset = event
            fig.add_shape(
                type="rect",
                x0=onset,
                x1=offset,
                y0=min(fig.data[0]['y']),  # Use trace min amplitude
                y1=max(fig.data[0]['y']),  # Use trace max amplitude
                fillcolor=color,
                opacity=0.4,
                line=dict(width=0)
            )
            fig.add_trace(go.Scatter(
                x=[onset, offset],
                y=[max(fig.data[0]['y']), max(fig.data[0]['y'])],
                mode='markers',
                marker=dict(color=color, size=6),
                name=name,
                showlegend=False
            ))

def plot_peaks_roi(fig, events, roi_amplitude, roi_time, name, color):
    """
    Adds markers for maximum amplitude peaks within fast_only events for a specific ROI.
    
    Parameters:
    - fig: Plotly figure to modify.
    - events: Array of event onset-offset pairs.
    - roi_amplitude: Array of amplitude values for the ROI.
    - roi_time: Array of time values corresponding to the ROI.
    - name: Name of the event type (e.g., 'Fast Only').
    - color: Color for the peak markers.
    """
    # Add a single legend entry for all max peaks
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(color=color, size=10, symbol='circle'),
        name=f'{name} Max Peak'
    ))

    for event in events:
        start_time, end_time = event

        # Find indices corresponding to the start and end times
        start_index = np.searchsorted(roi_time, start_time)
        end_index = np.searchsorted(roi_time, end_time)

        # Extract the relevant segment of the amplitude array
        segment_amplitude = roi_amplitude[start_index:end_index + 1]
        segment_time = roi_time[start_index:end_index + 1]

        # Find the maximum amplitude and its corresponding time
        max_amplitude = np.max(segment_amplitude)
        max_time = segment_time[np.argmax(segment_amplitude)]

        # Add a marker for the maximum peak without a legend entry
        fig.add_trace(go.Scatter(
            x=[max_time],
            y=[max_amplitude],
            mode='markers',
            marker=dict(
                color='white',  # Set the fill color to white (or transparent if supported)
                size=5,         # Circle size
                symbol='circle',  # Circle symbol
                line=dict(
                    color=color,  # Border color
                    width=2       # Border width
                )
            ),
            showlegend=False
        ))



# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
