from math import log
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import altair as alt
import pycbc
from pycbc.waveform import get_td_waveform, fd_approximants
import pylab
import openpyxl
import requests

#other imports
import matplotlib.pyplot as plt
from scipy import signal, datasets, fft
from scipy.signal import get_window
import gwpy

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwosc.datasets import event_gps
from gwosc.api import fetch_event_json

from copy import deepcopy
import base64

# Use the non-interactive Agg backend, which is recommended as a
# thread-safe backend.
# See https://matplotlib.org/3.3.2/faq/howto_faq.html#working-with-threads.
import matplotlib as mpl                
mpl.use("agg")

##############################################################################
# Workaround for the limited multi-threading support in matplotlib.
# Per the docs, we will avoid using `matplotlib.pyplot` for figures:
# https://matplotlib.org/3.3.2/faq/howto_faq.html#how-to-use-matplotlib-in-a-web-application-server.
# Moreover, we will guard all operations on the figure instances by the
# class-level lock in the Agg backend.
##############################################################################
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock


# --Set page config
apptitle = 'GWTC Global View'
st.set_page_config(page_title=apptitle, layout="wide")

#Title the app
st.title('Gravitational-wave Transient Catalog Dashboard')

#Top Dashboard Elements
st.markdown('### Metrics')

# Fetch the data from the URL and load it into a DataFrame
@st.cache_data
def load_and_group_data():
    df = pd.read_csv("https://gwosc.org/eventapi/csv/allevents/")
    
    catalogs = ['GWTC', 'GWTC-1-confident', 'GWTC-2', 'GWTC-2.1-confident', 'GWTC-3-confident', 'O3_Discovery_Papers']
    
    grouped_data = {}
    
    for catalog in catalogs:
        if catalog == 'GWTC':
            event_data = df[df['catalog.shortName'].str.contains('confident', case=False)]
        else:
            event_data = df[(df['catalog.shortName'] == catalog)]
        grouped_data[catalog] = event_data

    return grouped_data

grouped_data = load_and_group_data()

#create top row columns for selectbox and charts
col1, col2, col3 = st.columns(3)

with col1:
    selected_event = st.selectbox('Select Event (Defaults to GWTC)', grouped_data.keys())
    if selected_event in grouped_data:
        event_df = grouped_data[selected_event]

col1.write('Select an Event Catalog to learn more about each individual release')

#fix missing mass issue
event_df['total_mass_source'] = event_df['mass_1_source'] + event_df['mass_2_source']

event_df.to_excel('updated_GWTC.xlsx', index=False)

updated_excel = 'updated_GWTC.xlsx'

# Loads df to use for the rest of the dash
df = pd.read_excel(updated_excel)

# Count for total observations 
count = event_df.commonName.unique().size

col2.metric(label="Total Observations in this Catalog",
    value=(count),    
)
col2.write('The observations shown contain information for the mass of both sources')

# Sort mass for event type distribution
def categorize_event(row):
    if row['mass_1_source'] < 3 and row['mass_2_source'] < 3:
        return 'Binary Neutron Star'
    elif row['mass_1_source'] >= 3 and row['mass_2_source'] >= 3:
        return 'Binary Black Hole'
    else:
        return 'Neutron Star Black Hole'

# sourcery skip: assign-if-exp
df['Event'] = df.apply(categorize_event, axis=1)

# Group data by event type and count occurrences
grouped_df = df.groupby('Event').size().reset_index(name='Count')

# Create the pie chart
pie_chart = alt.Chart(grouped_df).mark_arc().encode(
    theta=alt.Theta(field='Count', type='quantitative'),
    color=alt.Color(field='Event', type='nominal'),
    tooltip=['Event', 'Count']
).properties(
    width=300,
    height=300,
    title='Merger Type Distribution'
)

col3.altair_chart(pie_chart, use_container_width=True)
expdr = col3.expander('Breakdown of the type of events we have detected so far')
expdr.write(
    'BBH are Binary Black Hole Mergers, BNS are Binary Neutron Star Mergers and NSBH is a merger between a black hole and a neutron star')

#mass chart for Dashboard
mass_chart = alt.Chart(df, title="Total Mass Histogram").mark_bar().encode(
    x=alt.X('total_mass_source:N', title='Total Mass', bin=True),
    y=alt.Y('count()', title='Count'),
    #tooltip=['commonName', 'GPS']
)

#Histogram for SNR
snr = alt.Chart(df, title="Network SNR Histogram").mark_bar().encode(
    x=alt.X('network_matched_filter_snr:Q', title='SNR', bin=True),
    y=alt.Y('count()', title='Count')
)

#Histogram for Distance
dist = alt.Chart(df, title="Distance Histogram").mark_bar().encode(
    x=alt.X('luminosity_distance:Q', title='Distance in Mpc', bin=alt.Bin(maxbins=10)),
    y=alt.Y('count()', title='Count')
)

#SECOND ROW COLUMNS
col4, col5, col6 = st.columns(3)

col4.altair_chart(mass_chart, use_container_width=True)
expdr = col4.expander('Show more info in column!')
expdr.write('More info!')

col5.altair_chart(dist, use_container_width=True)
expdr = col5.expander('Show more info in column!')
expdr.write('More info!')

col6.altair_chart(snr, use_container_width=True)
expdr = col6.expander('Show more info in column!')
expdr.write('This network SNR is the quadrature sum of the individual detector SNRs for all detectors involved in the reported trigger; ')
#cite from https://journals.aps.org/prx/pdf/10.1103/PhysRevX.9.031040

st.markdown('### Select an event to learn more')
#MAIN CHART FOR USER INPUT
event_chart = px.scatter(event_df, x="mass_1_source", y="mass_2_source", color="network_matched_filter_snr", labels={
    "network_matched_filter_snr": "Network SNR",
    "commonName": "Name",
    "mass_1_source": "Mass 1",
    "mass_2_source": "Mass 2", 
}, title= "Event Catalog of source-frame component masses m<sub>(i)</sub>", color_continuous_scale = "dense", hover_data=["commonName"])

event_chart.update_traces(
    marker=dict(size=10,
    symbol="circle",
    )
)
event_chart.update_layout(
    hovermode='x unified',
    width=900,
    height=450,
    xaxis_title="Mass 1 (M<sub>☉</sub>)",  # Add the smaller Solar Mass symbol using <sub> tag
    yaxis_title="Mass 2 (M<sub>☉</sub>)", 
    hoverdistance=-1,
)
event_chart.update_xaxes(
    title_standoff=10,
    title_font = {"size": 15},
)
event_chart.update_yaxes(
    title_standoff=10,
    title_font = {"size": 15},
)

# Function to filter event options based on input prefix
def filter_event_options(prefix):
    return event_df[event_df['commonName'].str.startswith(prefix)]['commonName'].tolist()

event_input = st.multiselect(
    "If you have an Event Name enter it below, otherwise click on an event in the chart to populate more information about that specific Event:",
    filter_event_options(""),
    default=[],
    key="event_input",
)
# Initialize select_event as an empty list
select_event = []
#User Selection
select_event = plotly_events(event_chart, click_event=True)

expander = st.expander("Expand for more information regarding the Event Catalog Chart")
expander.write('Info about the Chart')            

if event_input:
    selected_event_name = event_input[0]
    selected_event_row = df[df['commonName'] == selected_event_name]
    if not selected_event_row.empty:
        selected_x = selected_event_row['mass_1_source'].values[0]
        selected_y = selected_event_row['mass_2_source'].values[0]
        select_event = [{'x': selected_x, 'y': selected_y}]
    else:
        st.write("Selected event not found in the dataset.")

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
            
#CHARTS WITH USER INPUT
if select_event:
    #generate waveform
    hp, hc = get_td_waveform(approximant="IMRPhenomD",
                             mass1=mass_1,
                             mass2=mass_2,
                             delta_t=1.0/16384,
                             f_lower=45,
                             distance=dist)

    #Zoom in near the merger time
    wave = plt.figure(figsize=pylab.figaspect(0.4))
    plt.plot(hp.sample_times, hp, label='Plus Polarization')
    plt.plot(hp.sample_times, hc, label='Cross Polarization')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.xlim(-.5, .5)
    plt.legend()
    plt.grid()

    ##Gauge Indicators
    total_mass = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = total_mass_source,
    number = {"suffix": "(M<sub>☉</sub>)"},
    title = {'text': "Total Mass (M<sub>☉</sub>)"},
    gauge = {'axis': {'range': [None, 200]},
             'bar': {'color': "lightskyblue"},
             'bgcolor': "white",
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 181}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))

    total_mass.update_layout(
        autosize=False,
        width=300,
        height=300,
    )
    lum_dist = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = dist,
    number = {"suffix": "(Mpc)"},
    title = {'text': "Luminosity Distance <sub>(Mpc)</sub>"},
    gauge = {'axis': {'range': [None, 10000]},
             'bar': {'color': "lightskyblue"},
             'bgcolor': "white",
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 8280}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))

    lum_dist.update_layout(
        autosize=False,
        width=300,
        height=300,
    )

    m1 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = mass_1,
    number = {"suffix": "(M<sub>☉</sub>)"},
    title = {'text': "Mass of source 1 (M<sub>☉</sub>)"},
    gauge = {'axis': {'range': [None, 200]},
             'bar': {'color': "lightskyblue"},
             'bgcolor': "white",
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 105}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))

    m1.update_layout(
        autosize=False,
        width=300,
        height=300,
    )
    #gauge for mass2
    m2 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = mass_2,
    number = {"suffix": "(M<sub>☉</sub>)"},
    title = {'text': "Mass of source 2 (M<sub>☉</sub>)"},
    gauge = {'axis': {'range': [None, 200]},
             'bar': {'color': "lightskyblue"},
             'bgcolor': "white",
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 76}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))

    m2.update_layout(
        autosize=False,
        width=300,
        height=300,
    )

    #gauge for snr
    snr = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = snr, 
    title = {'text': "Network Matched Filter SNR"},
    gauge = {'axis': {'range': [None, 40]},
             'bar': {'color': "lightskyblue"},
             'bgcolor': "white",
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 33}},      
    ))
    snr.update_layout(
    autosize=False,
    width=300,
    height=300,
    )

    #Columns for Gauges
    col7, col8, col9 = st.columns(3)

    col7.write(total_mass)

    col8.write(m1)

    col9.write(m2)

    expander = st.expander("The total mass and the mass for each source, the red line indicates the largest mass found to date for each category, the event you selected is: " + event_name)
    expander.write("This shows the total mass of the source and the breakdown of the masses of the two components labeled as Mass 1 and Mass 2")

    col10, col11, = st.columns(2)

    col10.write(lum_dist)
    expdr = col10.expander('Luminosity Distance is how far, in Mpc, the merger is from the sun. This is a good indicator of the distance between the two sources')
    expdr.write('More info!')

    col11.write(snr)
    expdr = col11.expander('The Network Matched Filter SNR is a measure of the quality of the data, with a higher SNR giving us better data')
    expdr.write('More info!')

    #have users select a detector
    detectorlist = ['H1', 'L1', 'V1']
    detector = st.selectbox("Select a Detector, (Note: Not all events available for all detectors)", detectorlist)
    ## need to update to prevent error if detector is not available 

    #get timeseries and gps info to confirm
    segment = (int(gps_info)-5, int(gps_info)+5)
    ldata = TimeSeries.fetch_open_data(detector, *segment, verbose=True, cache=True)

    ###TEST FOR QTRANS
    st.subheader('Q-transform')

    hq = None
    chirp_mass = event_df['chirp_mass_source']
    bns = chirp_mass < 5
    t0 = datasets.event_gps(event_name)
    q_center = 100*(1/chirp_mass)
    q_center[q_center > 5] = 5
    qrange_min = float((q_center*0.8).min())
    qrange_max = float((q_center*1.2).max())
    qrange = (qrange_min, qrange_max)

    if bns.any():
        dt = 2
    else: 
        dt = 0.3

    hq = ldata.q_transform(outseg=(t0-dt, t0+dt), qrange=qrange)

    fig4 = hq.plot()
    ax = fig4.gca()
    fig4.colorbar(label="Normalised energy", vmax=25, vmin=0)
    ax.grid(False)
    ax.set_yscale('log')
    ax.set_ylim(ymin=20, ymax=1024)
    st.pyplot(fig4, clear_figure=True)

    #Spectrogram to confirm data is feeding through
    specgram = ldata.spectrogram(2, fftlength=1, overlap=.5)**(1/2.)

    plot = specgram.imshow(norm='log', vmin=5e-24, vmax=1e-19)
    ax = plot.gca()
    ax.set_yscale('log')
    ax.set_ylim(10, 2000)
    ax.colorbar(
        label=r'Gravitational-wave amplitude'
    )

    col12, col13 = st.columns(2)

    col12.write(wave)

    col13.pyplot(plot)

else:
    st.write("Click on event to view more details")




#--Add

#--Fix errors
#Detector Error
