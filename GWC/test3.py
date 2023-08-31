import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
#import django
#django.setup()
from event.models import Event
from tile.models import TileDataset
from parameter.models import Parameter
#from catalog.api_classbased import get_strain_files
from gwpy.timeseries import TimeSeries



import streamlit as st
import plotly.express as px

import streamlit as st
import plotly.express as px

# Assuming you have imported necessary functions and defined df

# Function to filter event options based on input prefix
def filter_event_options(prefix):
    return df[df['commonName'].str.startswith(prefix)]['commonName'].tolist()

# User input for event name autocomplete
event_prefix = st.sidebar.text_input("Enter event name prefix:", "").strip()
event_options = filter_event_options(event_prefix)
selected_event = st.sidebar.selectbox("Select event:", event_options)

# Automatically trigger the selection process if an event is selected
if selected_event:
    # Find the row in the DataFrame that matches the selected event
    selected_row = df[df["commonName"] == selected_event]

    if not selected_row.empty:
        st.markdown('### Selected Event: ' + selected_event)
        gps_info = event_gps(selected_event)
        if gps_info:
            st.write("GPS Information:", gps_info)
            mass_1 = selected_row['mass_1_source'].values[0]
            mass_2 = selected_row['mass_2_source'].values[0]
            dist = selected_row['luminosity_distance'].values[0]
            total_mass_source = selected_row['total_mass_source'].values[0]
            snr = selected_row['network_matched_filter_snr'].values[0]
        else:
            st.write("GPS Information not available for the selected event.")

# MAIN CHART FOR USER INPUT
event_chart = px.scatter(df, x="mass_1_source", y="mass_2_source", color="network_matched_filter_snr", labels={
    "network_matched_filter_snr": "Network SNR",
    "commonName": "Name",
    "mass_1_source": "Mass 1",
    "mass_2_source": "Mass 2", 
}, title= "Event Catalog of source-frame component masses m<sub>(i)</sub>", color_continuous_scale = "dense", hover_data=["commonName"])

# User Selection for plotly_events
select_event = plotly_events(event_chart, click_event=True)

if select_event:
    # Retrieve clicked x and y values
    clicked_x = select_event[0]['x']
    clicked_y = select_event[0]['y']

    # Find the row in the DataFrame that matches the clicked x and y values
    selected_row = df[(df["mass_1_source"] == clicked_x) & (df["mass_2_source"] == clicked_y)]

    if not selected_row.empty:
        selected_common_name = selected_row["commonName"].values[0]
        event_name = selected_common_name
        st.markdown('### Selected Event: ' + event_name)
        gps_info = event_gps(event_name)
        if gps_info:
            st.write("GPS Information:", gps_info)
            mass_1 = selected_row['mass_1_source'].values[0]
            mass_2 = selected_row['mass_2_source'].values[0]
            dist = selected_row['luminosity_distance'].values[0]
            total_mass_source = selected_row['total_mass_source'].values[0]
            snr = selected_row['network_matched_filter_snr'].values[0]
        else:
            st.write("GPS Information not available for the selected event.")

# Display the chart
st.plotly_chart(event_chart)
