from math import log
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
import plotly.graph_objects as go
import altair as alt
import pycbc
from pycbc.waveform import get_td_waveform
import pylab
import openpyxl
import requests
import plotly.io as pio
from urllib.parse import parse_qs, urlparse
from matplotlib.ticker import FuncFormatter, AutoLocator
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
from scipy.io import wavfile
import io
from astropy.time import Time

# --Set page config
apptitle = 'GWTC Global View'
st.set_page_config(page_title=apptitle, layout="wide")

#Title the app
st.title('Gravitational-wave Catalog Dashboard')
st.write('The Gravitational-wave Catalog is a cumulative set of gravitational wave transients maintained by the LIGO/Virgo/KAGRA collaboration. The online catalog contains confidently-detected events from multiple data releases. For further information, please visit https://gwosc.org')
st.write(    
    """
    Catalogs:  
    - General Catalogs like :red[GWTC] are cumulative catalogs describing all the gravitational-wave transients found in an observing run.
    - Catalogs labeled as :red[Confident] have strain, segments, search results and parameter estimation values with corresponding 90 percent credible intervals.
    - :red[Discovery Papers] contain notable events in the respective run. 
    - Other Catalogs such as :red[Marginal, Preliminary or Auxillary], can be found here: https://gwosc.org/eventapi/html/
    """
        )
# Fetch the data from the URL and load it into a DataFrame
@st.cache_data
def load_and_group_data():
    df = pd.read_csv("https://gwosc.org/eventapi/csv/allevents/")

    # Substrings to exclude
    exclude_substrings = ['marginal', 'auxiliary', 'preliminary', 'initial']

    # Filter out events based on excluded substrings
    df_filtered = df[~df['catalog.shortName'].str.lower().str.contains('|'.join(exclude_substrings))]

    # Get unique catalogs after exclusion
    catalogs = df_filtered['catalog.shortName'].unique()

    # Create a dictionary to store grouped data
    grouped_data = {}

    gwtc_catalogs = [cat for cat in catalogs if 'GWTC' in cat]

    # If there's more than one "GWTC" catalog, create a default catalog containing all relevant names
    if len(gwtc_catalogs) > 1:
        default_catalog = 'GWTC'
        grouped_data[default_catalog] = pd.concat([df_filtered[df_filtered['catalog.shortName'] == cat] for cat in gwtc_catalogs])
    else:
        default_catalog = gwtc_catalogs[0] if gwtc_catalogs else None  # Handle the case where no "GWTC" catalog is found

    # Create subgroups for each catalog
    for cat in catalogs:
        if cat != default_catalog:
            grouped_data[cat] = df_filtered[df_filtered['catalog.shortName'] == cat]

    return grouped_data, default_catalog

grouped_data, default_catalog = load_and_group_data()

st.divider()

## Create top row columns for selectbox and charts
col1, col2, col3 = st.columns(3)

with col1:
    selected_cat = st.selectbox('Select an Event Catalog (Defaults to GWTC)', grouped_data.keys())
    if selected_cat in grouped_data:
        event_df = grouped_data[selected_cat]

col1.write('Each catalog contains a collection of events observed during a LIGO/Virgo observation run. This Dashboard uses the following parameters: Total Mass, Mass 1, Mass 2, SNR and Luminosity Distance. For a full list of parameters in the catalog, select an event to populate a breakdown at the bottom of the page. ')

# Eliminate rows with missing mass_1_source or mass_2_source
event_df = event_df.dropna(subset=['mass_1_source', 'mass_2_source'])
event_df['total_mass_source'] = event_df['mass_1_source'] + event_df['mass_2_source'] #fixes missing mass issue
event_df.to_excel('updated_GWTC.xlsx', index=False)
updated_excel = 'updated_GWTC.xlsx'

# Loads df to use for the rest of the dash
df = pd.read_excel(updated_excel)

#max values
max_mass = df['total_mass_source'].max()
max_mass_1 = df['mass_1_source'].max()
max_mass_2 = df['mass_2_source'].max()
max_snr = df['network_matched_filter_snr'].max()
max_lum = df['luminosity_distance'].max()
event_max_mass = df.loc[df['total_mass_source'].idxmax(), 'commonName']
event_max_mass_1 = df.loc[df['mass_1_source'].idxmax(), 'commonName']
event_max_mass_2 = df.loc[df['mass_2_source'].idxmax(), 'commonName']
if all(x == event_max_mass for x in [event_max_mass_1, event_max_mass_2]):
    event_max = event_max_mass

# Count for total observations 
count = event_df.commonName.unique().size

## Continue columns
col2.metric(label="Total Observations in this Catalog",
    value=(count),    
)
col2.write('This is the number of confident observations for the catalog selected, for a complete list of all events please visit: https://gwosc.org/eventapi/' )

df_events = df.drop_duplicates(subset=['commonName'])
# Sort mass for event type distribution
def categorize_event(row):
    if row['mass_1_source'] < 3 and row['mass_2_source'] < 3:
        return 'Binary Neutron Star'
    elif row['mass_1_source'] >= 3 and row['mass_2_source'] >= 3:
        return 'Binary Black Hole'
    else:
        return 'Neutron Star - Black Hole'
df_events['Event'] = df_events.apply(categorize_event, axis=1)

# Group data by event type and count occurrences
grouped_df = df_events.groupby('Event').size().reset_index(name='Count')

# Custom color scale for events
event_colors = alt.Scale(
    domain=['Binary Black Hole', 'Neutron Star - Black Hole', 'Binary Neutron Star'],
    range=['#201e66', '#504eca', '#bdbceb']  
)
# Pie chart
pie_chart = alt.Chart(grouped_df).mark_arc().encode(
    theta=alt.Theta(field='Count', type='quantitative'),
    color=alt.Color(field='Event', type='nominal', scale=event_colors),
    tooltip=['Event', 'Count']
).properties(
    width=300,
    height=300,
    title='Merger Type Distribution'
)
col3.altair_chart(pie_chart, use_container_width=True)
col3.write('The observed events are mergers of neutron stars and/or black holes.')
st.divider()    

# Mass chart for Dashboard
mass_chart = alt.Chart(df, title="Total Mass Histogram in Solar Masses").mark_bar().encode(
    x=alt.X('total_mass_source:Q', title='Total Source Frame Mass', bin=True),
    y=alt.Y('count()', title='Count'),
    #tooltip=['commonName', 'GPS']
)
# Histogram for Distance
dist = alt.Chart(df, title="Luminosity Distance Histogram").mark_bar().encode(
    x=alt.X('luminosity_distance:Q', title='Distance in Mpc', bin=alt.Bin(maxbins=10)),
    y=alt.Y('count()', title='Count')
)
# Histogram for SNR
snr = alt.Chart(df, title="Network SNR Histogram").mark_bar().encode(
    x=alt.X('network_matched_filter_snr:Q', title='SNR', bin=True),
    y=alt.Y('count()', title='Count')
)

#SECOND ROW COLUMNS
col4, col5, col6 = st.columns(3)
col4.altair_chart(mass_chart, use_container_width=True)
col4.write('Shows the distribution of mass for objects contained in the Catalog selected.')
col5.altair_chart(dist, use_container_width=True)
col5.write('Shows the distribution of luminosity distance in megaparsec (3.26 million lightyears) for objects contained in the Catalog selected.')
col6.altair_chart(snr, use_container_width=True)
col6.write('This network signal to noise ratio (SNR) is the quadrature sum of the individual detector SNRs for all detectors involved in the reported trigger. ')
st.divider()
st.markdown('### Select an event from the Catalog: ' + selected_cat)
#MAIN CHART FOR USER INPUT
event_chart = px.scatter(df_events, x="mass_1_source", y="mass_2_source", color="network_matched_filter_snr", labels={
    "network_matched_filter_snr": "Network SNR",
    "luminosity_distance": "Luminosity Distance (Mpc)",
    "commonName": "Name",
    "mass_1_source": "Mass 1",
    "mass_2_source": "Mass 2", 
}, title= "Event Catalog of source-frame component masses (M<sub>☉</sub>)", color_continuous_scale = "dense", hover_data=["commonName"])

event_chart.update_traces(
    marker=dict(size=10,
    symbol="circle",
    )
)
event_chart.update_layout(
    hovermode='x unified',
    width=900,
    height=450,
    xaxis_title="Source Frame Mass 1 (M<sub>☉</sub>)", 
    yaxis_title="Source Frame Mass 2 (M<sub>☉</sub>)", 
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

def filter_event_options(prefix):
    return df[df['commonName'].str.startswith(prefix)]['commonName'].tolist()
event_options = filter_event_options("")

event_url = st.query_params.get("event_name", "")

# Create the selectbox with options
event_input = st.selectbox(
    "Select an event from the drop-down list:",
    [""] + event_options,
    key="event_input",
)

st.write("OR click on an event in the chart. :red[**Clear drop down menu to enable chart functionality]")
select_event = plotly_events(event_chart, click_event=True)

if not event_input and select_event:
    selected_event_row = df[(df['mass_1_source'] == select_event[0]['x']) & (df['mass_2_source'] == select_event[0]['y'])]
    if not selected_event_row.empty:
        event_input = selected_event_row['commonName'].values[0]
        event_url = select_event
elif event_url and not event_input:
    event_input = event_url

if event_input:
    selected_event_row = df[df['commonName'] == event_input]
    if not selected_event_row.empty:
        selected_x = selected_event_row['mass_1_source'].values[0]
        selected_y = selected_event_row['mass_2_source'].values[0]
        select_event = [{'x': selected_x, 'y': selected_y}]

with st.expander(label="The chart allows the following interactivity: ", expanded=True):
    st.write(
    """
    - Pan and Zoom
    - Box Selection
    - Download chart as a PNG
    """
    
)
## USER INPUT OPTIONS
if event_input:
    selected_event_name = event_input
    selected_event_row = df[df['commonName'] == selected_event_name]
    if not selected_event_row.empty:
        selected_x = selected_event_row['mass_1_source'].values[0]
        selected_y = selected_event_row['mass_2_source'].values[0]
        select_event = [{'x': selected_x, 'y': selected_y}]
    else:
        selected_event_name = ("Click on an Event")
    if select_event:
        clicked_x = select_event[0]['x']
        clicked_y = select_event[0]['y']
        selected_row = df[(df["mass_1_source"] == clicked_x) & (df["mass_2_source"] == clicked_y)]
        if not selected_row.empty:
            event_name = selected_row["commonName"].values[0]
            if gps_info := event_gps(event_name):
                new_df = selected_event_row.copy()
                mass_1 = selected_row['mass_1_source'].values[0]
                mass_2 = selected_row['mass_2_source'].values[0]
                dist = selected_row['luminosity_distance'].values[0]
                total_mass_source = selected_row['total_mass_source'].values[0]
                snr = selected_row['network_matched_filter_snr'].values[0]
                chirp = selected_row['chirp_mass'].values[0]
                
        gps_time = gps_info

        utc_datetime = Time(gps_time, format='gps').datetime
        datetime = utc_datetime.strftime('%Y-%m-%d %H:%M:%S')          
else:
    st.query_params["event_name"] = event_name
    
if select_event or event_input:  
    st.markdown('### Selected Event: ' + event_name)
    st.write('GPS Time:', + gps_info, " is the end time or merger time of the event in GPS seconds.")
    st.write('This GPS time corresponds to the date: ', datetime)
    with st.expander(label="Legend for Gauges: ", expanded=True):
        st.write('The :blue[[blue area]] indicates the margin of error for each source.')
        st.write('The :red[red line |] indicates the largest value in the catalog selected')
        st.write('$M_{\odot}$ : Solar mass is $1.9891x10^{30}$ kg')
        st.write(    
    """
    Parameters:
    - Total Mass
    - Mass of Source 1
    - Mass of Source 2
    - Luminosity Distance
    - Network SNR
    """
        )
        st.write('*Note: Some events may not have error information.')
else:
    selected_event_name = ("Click on an Event")
            
st.divider()

## CHARTS WITH USER INPUT
if select_event or event_input:    
    ##Gauge Indicators
    total_mass_lower = selected_row['total_mass_source_lower'].values[0] + total_mass_source
    total_mass_upper = selected_row['total_mass_source_upper'].values[0] + total_mass_source    
    total_mass = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = total_mass_source,
    number = {"suffix": "M<sub>☉</sub>"},
    title = {'text': "Total Mass (M<sub>☉</sub>)"},
    gauge = {'axis': {'range': [None, 250]},
            'bar': {'color': "#4751a5"},             
            'steps' : [
                {'range': [total_mass_source, total_mass_upper], 'color': "lightskyblue"},
                {'range': [total_mass_source, total_mass_lower], 'color': "lightskyblue"}],             
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_mass}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    total_mass.update_layout(
        autosize=False,
        width=400,
        height=400,
    )
    total_mass.add_annotation(
    x=0.5,  
    y=-0.05,  
    text=f'Max in catalog: {max_mass} M<sub>☉</sub>',
    showarrow=False,
    font=dict(size=15, color='red')
)

    #mass 1 gauge
    m1_lower = selected_row['mass_1_source_lower'].values[0] + selected_row['mass_1_source'].values[0] 
    m1_upper = selected_row['mass_1_source_upper'].values[0] + selected_row['mass_1_source'].values[0]    
    m1 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = mass_1,
    number = {"suffix": "M<sub>☉</sub>"},
    title = {'text': "Mass of source 1 (M<sub>☉</sub>)"},
    gauge = {'axis': {'range': [None, 200]},
            'bar': {'color': "#4751a5"},             
            'steps' : [
                {'range': [mass_1, m1_upper], 'color': "lightskyblue"},
                {'range': [mass_1, m1_lower], 'color': "lightskyblue"}],             
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_mass_1, }},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    m1.update_layout(
        autosize=False,
        width=400,
        height=400,
    )
    m1.add_annotation(
    x=0.5,
    y=-0.05,  
    text=f'Max in catalog: {max_mass_1} M<sub>☉</sub>',
    showarrow=False,
    font=dict(size=15, color='red')
) 
    #mass 2 gauge
    m2_lower = selected_row['mass_2_source_lower'].values[0] + selected_row['mass_2_source'].values[0] 
    m2_upper = selected_row['mass_2_source_upper'].values[0] + selected_row['mass_2_source'].values[0]    
    m2 = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = mass_2,
    number = {"suffix": "M<sub>☉</sub>"},
    title = {'text': "Mass of source 2 (M<sub>☉</sub>)"},
    gauge = {'axis': {'range': [None, 180]},  
            'bar': {'color': "#4751a5"},         
            'steps' : [
                {'range': [mass_2, m2_upper], 'color': "lightskyblue"},
                {'range': [mass_2, m2_lower], 'color': "lightskyblue"}],
            'bgcolor': "white",           
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_mass_2}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    m2.update_layout(
        autosize=False,
        width=400,
        height=400,
    )
    m2.add_annotation(
    x=0.5,  
    y=-.05,  
    text=f'Max in catalog: {max_mass_2} M<sub>☉</sub>',
    showarrow=False,
    font=dict(size=15, color='red')
)
    #lum dist gauge
    lum_dist_lower = selected_row['luminosity_distance_lower'].values[0] + selected_row['luminosity_distance'].values[0] 
    lum_dist_upper = selected_row['luminosity_distance_upper'].values[0] + selected_row['luminosity_distance'].values[0]        
    #Convert lum_dist from Mpc to Gpc 
    dist = dist/1000
    lum_dist_lower = lum_dist_lower/1000
    lum_dist_upper = lum_dist_upper/1000 
    lum_max = max_lum/1000
    lum_dist = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = dist,
    number = {"suffix": "Gpc"},
    title = {'text': "Luminosity Distance (Gpc)"},
    gauge = {'axis': {'range': [None, 18]},
            'bar': {'color': "#4751a5"},
            'steps' : [
                {'range': [dist, lum_dist_upper], 'color': "lightskyblue"},
                {'range': [dist, lum_dist_lower], 'color': "lightskyblue"}],             
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': lum_max}},
    domain = {'x': [0, 1], 'y': [0, 1]}
    ))
    lum_dist.update_layout(
        autosize=False,
        width=400,
        height=400,
        xaxis_title= lum_dist_lower + lum_dist_upper, 
    )
    
    lum_dist.add_annotation(
    x=0.5,  
    y=-0.01,  
    text=f'Max in catalog: {lum_max} Gpc',
    showarrow=False,
    font=dict(size=15, color='red')
)
    #snr gauge
    snr_lower = selected_row['network_matched_filter_snr_lower'].values[0] + selected_row['network_matched_filter_snr'].values[0] 
    snr_upper = selected_row['network_matched_filter_snr_upper'].values[0] + selected_row['network_matched_filter_snr'].values[0]
    snr_chart = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = snr, 
    title = {'text': "Network Matched Filter SNR"},
    gauge = {'axis': {'range': [None, 40]},
            'steps' : [
                {'range': [snr, snr_upper], 'color': "lightskyblue"},
                {'range': [snr, snr_lower], 'color': "lightskyblue"}],
            'bar': {'color': "#4751a5"},
            'bgcolor': "white",
            'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': max_snr}},      
    ))
    snr_chart.update_layout(
    autosize=False,
    width=400,
    height=400,
    )
    snr_chart.add_annotation(
    x=0.5,  
    y=0,  
    text=f'Max in catalog: {max_snr}',
    showarrow=False,
    font=dict(size=15, color='red')
)
    st.subheader("Selected event " + event_name)
## RIDGE LINE PLOTS
    #total mass
    ridge_mass = go.Figure()
    ridge_mass.add_trace(go.Violin(x=df['total_mass_source'], line_color='#808080', name = ''))
    ridge_mass.add_shape(
        dict(
            type="line",
            x0=total_mass_source,
            x1=total_mass_source,
            y0=0,
            y1=2,  
            line=dict(color="#4751a5", width=3),
        )
    )
    ridge_mass.update_traces(orientation='h', side='positive', width=4, points=False, hovertemplate=None, hoverinfo='skip')
    ridge_mass.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    ridge_mass.update_layout(
        title = {'text': "Total Mass (M<sub>☉</sub>)"},
        autosize=False,
        width=400,
        height=300,
        annotations=[
            dict(
                text= "Distribution of mass in catalog.",
                xref="paper",
                yref="paper",
                x=0.5, 
                y=-0.5,  
                showarrow=False,
                font=dict(size=15))
        ]
    )
    
    ridge_mass.add_annotation(
    x=total_mass_source + 24,  
    y=2.2,  
    text=f'Mass: {total_mass_source:.2f} M<sub>☉</sub>',
    showarrow=False,
    font=dict(size=13, color='#504eca')
) 
    #mass1 plot
    ridge_mass1 = go.Figure()
    config = {'displayModeBar': False}
    ridge_mass1.add_trace(go.Violin(x=df['mass_1_source'], line_color='#808080', name = ''))
    ridge_mass1.add_shape(
        dict(
            type="line",
            x0=mass_1,
            x1=mass_1,
            y0=0,
            y1=2, 
            line=dict(color="#4751a5", width=3),
        )
    )
    ridge_mass1.update_traces(orientation='h', side='positive', width=4, points=False, hovertemplate=None, hoverinfo='skip')
    ridge_mass1.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    ridge_mass1.update_layout(
        autosize=False,
        title = {'text': "Mass(M<sub>☉</sub>) Source 1"},
        width=400,
        height=300,
        annotations=[
            dict(
                text= "Distribution of mass in catalog.",
                xref="paper",
                yref="paper",
                x=0.5,  
                y=-0.5,  
                showarrow=False,
                font=dict(size=15),
            )
        ]
    )
    
    ridge_mass1.add_annotation(
    x=mass_1 + 13,  
    y=2.2,  
    text=f'Mass: {mass_1} M<sub>☉</sub>',
    showarrow=False,
    font=dict(size=13, color='#504eca')
)
    #mass2 plot
    ridge_mass2 = go.Figure()
    config = {'displayModeBar': False}
    ridge_mass2.add_trace(go.Violin(x=df['mass_2_source'], line_color='#808080', name = ''))
    ridge_mass2.add_shape(
        dict(
            type="line",
            x0=mass_2,
            x1=mass_2,
            y0=0,
            y1=2, 
            line=dict(color="#4751a5", width=3),
        )
    )
    ridge_mass2.update_traces(orientation='h', side='positive', width=4, points=False, hovertemplate=None, hoverinfo='skip')
    ridge_mass2.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    ridge_mass2.update_layout(
        title = {'text': "Mass(M<sub>☉</sub>) Source 2"},
        autosize=False,
        width=400,
        height=300,
        annotations=[
            dict(
                text= "Distribution of mass in catalog",
                xref="paper",
                yref="paper",
                x=0.5,  
                y=-0.5, 
                showarrow=False,
                font=dict(size=15),
            )
        ]
    )
    ridge_mass2.add_annotation(
    x=mass_2 + 10,  
    y=2.2,  
    text=f'Mass: {mass_2} M<sub>☉</sub>',
    showarrow=False,
    font=dict(size=13, color='#504eca')
)

    #lum_dist  plot
    ridge_dist = go.Figure()
    config = {'displayModeBar': False}
    ridge_dist.add_trace(go.Violin(x=df['luminosity_distance'], line_color='#808080', name = ''))
    ridge_dist.add_shape(
        dict(
            type="line",
            x0=dist,
            x1=dist,
            y0=0,
            y1=2,  
            line=dict(color="#4751a5", width=3),
        )
    )
    ridge_dist.update_traces(orientation='h', side='positive', width=4, points=False, hovertemplate=None, hoverinfo='skip')
    ridge_dist.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    ridge_dist.update_layout(
        title = {'text': "Luminosity Distance (Gpc)"},
        autosize=False,
        width=400,
        height=300,
        annotations=[
            dict(
                text= "The luminosity distance in relation to the catalogs range in Gpc.",
                xref="paper",
                yref="paper",
                x=0.5,  
                y=-0.5, 
                showarrow=False,
                font=dict(size=15),
            )
        ]
    )
    ridge_dist.add_annotation(
    x=dist + 1400,  
    y=2.2,  
    text=f'Distance: {dist} Gpc',
    showarrow=False,
    font=dict(size=13, color='#504eca')
)
    #snr ridge plot
    ridge_snr = go.Figure()
    config = {'displayModeBar': False}
    ridge_snr.add_trace(go.Violin(x=df['network_matched_filter_snr'], line_color='#808080', name = ''))
    ridge_snr.add_shape(
        dict(
            type="line",
            x0=snr,
            x1=snr,
            y0=0,
            y1=2, 
            line=dict(color="#4751a5", width=3),
        )
    )
    ridge_snr.update_traces(orientation='h', side='positive', width=4, points=False, hovertemplate=None, hoverinfo='skip')
    ridge_snr.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    ridge_snr.update_layout(
        title = {'text': "Network Matched Filter SNR"},
        autosize=False,
        width=400,
        height=300,
        annotations=[
            dict(
                text= "The network SNR in relation to the catalogs distribution.",
                xref="paper",
                yref="paper",
                x=0.5,  
                y=-0.5, 
                showarrow=False,
                font=dict(size=15),
            )
        ]
    )
    ridge_snr.add_annotation(
    x=snr + 2,  
    y=2.2,  
    text=f'SNR: {snr}',
    showarrow=False,
    font=dict(size=13, color='#504eca')
)
    #Columns for Gauges
    col7, col8, col9 = st.columns(3)
    col7.write(total_mass)
    col7.write(ridge_mass)
    col8.write(m1)
    col8.write(ridge_mass1) 
    col9.write(m2)
    col9.write(ridge_mass2)
    st.divider()
    #second column
    st.subheader("Selected event: " + event_name)
    col10, col11, = st.columns(2)
    col10.write(lum_dist)
    col10.write(ridge_dist)
    col11.write(snr_chart)
    col11.write(ridge_snr)
    st.divider()
    #have users select a detector
    detectorlist = ['H1', 'L1', 'V1']
    detector = st.selectbox("Select a Detector, (Note: Not all events available for all detectors.)", detectorlist)

    #generate waveform
    hp, hc = get_td_waveform(approximant="IMRPhenomD",
                            mass1=mass_1,
                            mass2=mass_2,
                            delta_t=1.0/16384,
                            f_lower=45,
                            distance=dist)
    
    # Convert the TimeSeries data to a numpy array
    hp_array = np.array(hp)
    # Scale the data to 16-bit integer values
    hp_scaled = np.int16(hp_array / np.max(np.abs(hp_array)) * 32767)
    # Save the waveform as a WAV file
    wavfile.write("waveform.wav", 44100, hp_scaled)
    
    #Zoom in near the merger time
    wave = plt.figure(figsize=pylab.figaspect(0.4))
    plt.plot(hp.sample_times, hp, label='Plus Polarization')
    plt.plot(hp.sample_times, hc, label='Cross Polarization')
    plt.xlabel('Time (s)')
    plt.ylabel('Strain')
    plt.xlim(-.5, .5)
    plt.legend()
    plt.grid()
    
    # Fetch Time Series Data
    def fetch_time_series(detector, segment):
        try:
            return TimeSeries.fetch_open_data(detector, *segment, verbose=True, cache=True)
        except Exception as e:
            st.error(f"Please select a valid detector: {str(e)}")
            return None

    bns = False  # Initialize bns to a default value
    ldata = None  # Initialize ldata to None

    if gps_info:
        # Define the segment based on GPS info
        segment = (int(gps_info) - 5, int(gps_info) + 5)

        # Fetch time series data for the selected detector
        ldata = fetch_time_series(detector, segment)

    if ldata is not None:  # Check if ldata is not None
        chirp_mass = selected_row['chirp_mass_source'].values[0]
        if chirp_mass < 5:
            bns = True

        if bns:
            dt = 2
        else:
            dt = 0.3

        t0 = datasets.event_gps(event_name)
        q_center = 100 * (1 / chirp_mass)
        if q_center < 5:
            q_center = 5
        qrange = (int(q_center * 0.8), int(q_center * 1.2))
        outseg = (t0 - dt, t0 + dt)

        if ldata is not None:  # Check if ldata is not None again before using it
            hq = ldata.q_transform(outseg=outseg, qrange=qrange)
            x_values = hq.times.value - t0  # Calculate the time relative to t0
            fig4 = hq.plot()
            ax = fig4.gca()
            fig4.colorbar(label="Normalised energy", vmax=25, vmin=0)
            ax.grid(False)
            ax.set_yscale('log')
            ax.set_ylim(ymin=20, ymax=1024)

        #last column
        col12, col13 = st.columns(2)
        col12.subheader('Q-transform')            
        col12.pyplot(fig4, clear_figure=True)
        col12.write("""
        A Q-transform plot shows how a signal’s frequency changes with time.
        * The x-axis shows time
        * The y-axis shows frequency

        The color scale shows the amount of “energy” or “signal power” in each time-frequency pixel.
        
        """)
        col13.subheader('Waveform')
        col13.write(wave)
        col13.write('Listen to what the waveform sounds like')
        col13.audio("waveform.wav")
        col13.write('The waveform is a simplified example of the gravitational waveform radiated during a compact binary coalescence with the mass values of this source. ')
    else:
        st.write("Click on a event to view more details")
        
    def format_gps(val):
        gps_info = val["GPS"]
        return f"${gps_info:.2f}$"

    def format_total_mass_source(val):
        mass = val["total_mass_source"]
        lower = val["total_mass_source_lower"]
        upper = val["total_mass_source_upper"]
        return f"${mass:.2f}_{{{lower}}}^{{+{upper}}}$"
    def format_mass_1_source(val):
        mass_1 = val["mass_1_source"]
        lower = val["mass_1_source_lower"]
        upper = val["mass_1_source_upper"]
        return f"${mass_1:.2f}_{{{lower}}}^{{+{upper}}}$"
    def format_mass_2_source(val):
        mass_2 = val["mass_2_source"]
        lower = val["mass_2_source_lower"]
        upper = val["mass_2_source_upper"]
        return f"${mass_2:.2f}_{{{lower}}}^{{+{upper}}}$"

    def format_network_matched_filter_snr(val):
        snr = val["network_matched_filter_snr"]
        lower = val["network_matched_filter_snr_lower"]
        upper = val["network_matched_filter_snr_upper"]
        return f"${snr:.2f}_{{{lower}}}^{{+{upper}}}$"

    def format_lum_distance(val):
        lum = val["luminosity_distance"]
        lower = val["luminosity_distance_lower"]
        upper = val["luminosity_distance_upper"]
        return f"${lum:.1f}_{{{lower}}}^{{+{upper}}}$"
    
    def format_chi_eff(val):
        chi = val["chi_eff"]
        lower = val["chi_eff_lower"]
        upper = val["chi_eff_upper"]
        return f"${chi:.2f}_{{{lower}}}^{{+{upper}}}$"    

    def format_chirp_mass(val):
        chirp = val["chirp_mass_source"]
        lower = val["chirp_mass_source_lower"]
        upper = val["chirp_mass_source_upper"]
        return f"${chirp:.2f}_{{{lower}}}^{{+{upper}}}$"
    
    def format_redshift(val):
        red = val["redshift"]
        lower = val["redshift_lower"]
        upper = val["redshift_upper"]
        return f"${red:.2f}_{{{lower}}}^{{+{upper}}}$"
    
    def format_final_mass_source(val):
        final = val["final_mass_source"]
        lower = val["final_mass_source_lower"]
        upper = val["final_mass_source_upper"]
        return f"${final}_{{{lower}}}^{{+{upper}}}$"
    
    renamed_df = new_df.rename(columns={"id": "ID", "commonName": "Common Name", "version": "Version", "catalog.shortName": "Catalog Short Name", "reference": "Reference", "far": "FAR", "p_astro": "P Astro"})

    formatted_df = renamed_df.copy(deep=True)
    formatted_df = formatted_df.fillna('')
    formatted_df["GPS"] = formatted_df.apply(format_gps, axis=1)
    formatted_df["Total Mass Source"] = formatted_df.apply(format_total_mass_source, axis=1)
    formatted_df["Mass 1 Source"] = formatted_df.apply(format_mass_1_source, axis=1)
    formatted_df["Mass 2 Source"] = formatted_df.apply(format_mass_2_source, axis=1)
    formatted_df["Network Matched Filter SNR"] = formatted_df.apply(format_network_matched_filter_snr, axis=1)
    formatted_df["Luminosity Distance"] = formatted_df.apply(format_lum_distance, axis=1)
    formatted_df["Chi eff"] = formatted_df.apply(format_chi_eff, axis=1)
    formatted_df["Chirp Mass Source"] = formatted_df.apply(format_chirp_mass, axis=1)
    formatted_df["Redshift"] = formatted_df.apply(format_redshift, axis=1)
    formatted_df["Final Mass Source"] = formatted_df.apply(format_final_mass_source, axis=1)
    

    first_columns = ["ID", "Common Name", "Version", "Catalog Short Name", "GPS", "Reference", "jsonurl"]
    second_columns = ["Total Mass Source", "Mass 1 Source", "Mass 2 Source", "Network Matched Filter SNR"]
    third_columns = ["Luminosity Distance", "Chi eff", "Chirp Mass Source", "Redshift", "FAR", "P Astro", "Final Mass Source"]
    st.divider()
    st.subheader('Full catalog information for selected event: ' + selected_event_name)

    
    st.write(formatted_df[first_columns].to_markdown(index=False), unsafe_allow_html=True)
    st.write(formatted_df[second_columns].to_markdown(index=False), unsafe_allow_html=True)

    with st.expander(label="Parameters:", expanded=True):
        st.write(    
    """
    - Total Mass Source: The source-frame combined mass of the primary and secondary mass.
    - Mass 1 and Mass 2 Source: The source of the heavier and lighter objects in the binary, respectively.
    - Network SNR: The matched filter singal to noise raiot in the gravitational wave detector network.
    """
        )

    st.write(formatted_df[third_columns].to_markdown(index=False), unsafe_allow_html=True)

    with st.expander(label="Parameters:", expanded=True):
        st.write(    
    """
    - Luminosity Distance: The luminosity distance of the source. 
    - Chi Eff: The effective inspiral spin parameter. 
    - Chirp Mass Soruce: The source-frame chirp mass. 
    - Redshift: The redshift depending on specified cosmology. 
    - FAR: False alarm rate. 
    - P Astro: Likelihood that a detected event is from a genuine source. 
    - Final Mass Source: The source-frame remnant mass estimated using the spins evolved to the ISCO frequency. 
    """
        ) 
    st.divider()
st.subheader('GWTC-3: Compact Binary Coalescences Observed by LIGO and Virgo During the Second Part of the Third Observing Run')
st.write(' https://arxiv.org/abs/2111.03606')
st.divider()
st.header('About this app')
st.write('This app was made with the use of data or software obtained from the Gravitational Wave Open Science Center (gwosc.org), a service of the LIGO Scientific Collaboration, the Virgo Collaboration, and KAGRA.')
st.write('To learn more about Gravitational waves please visit the [Gravitational Wave Open Science Center Learning Path](https://gwosc.org/path/)')
st.write('GWOSC - https://gwosc.org')
st.divider()