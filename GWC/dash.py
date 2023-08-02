import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
from streamlit_elements import elements, mui, html
from streamlit_elements import dashboard
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import altair as alt

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
apptitle = 'GWD Global View'

st.set_page_config(page_title=apptitle, layout="wide")

#Title the app
st.title('GWD Dashboard')

# Fetch the data from the URL and load it into a DataFrame
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv("https://gwosc.org/eventapi/csv/GWTC/")

df = get_data()

#num of unique events
count = df.commonName.unique().size


#mass chart for Dashboard
mass_chart = alt.Chart(df, title="Mass 1 vs Mass 2").mark_circle().encode(
    x=alt.X('mass_1_source:Q', title='Source of Mass 1'),
    y=alt.Y('mass_2_source:Q', title='Source of Mass 2'),
    tooltip=['commonName', 'GPS']
)

#Histogram for SNR
snr = alt.Chart(df, title="Histogram of Network SNR").mark_bar().encode(
    x=alt.X('network_matched_filter_snr:Q', title='SNR', bin=True),
    y=alt.Y('count()', title='Count')
)

#Histogram for Distance
dist = alt.Chart(df, title="Histogram of Distance").mark_bar().encode(
    x=alt.X('luminosity_distance:Q', title='Distance in Mpc'),
    y=alt.Y('count()', title='Count')
)

#Top Dashboard Elements
st.markdown('### Metrics')

col1, col2, col3 = st.columns(3)

col1.metric(label="Total Observations to Date",
    value=(count),
)

col2.metric(
    label="Total Obvs Time",
    value=("test"),
)

col3.metric(
    label="Current Run",
    value=("O4") 
)

#second row of columns
col4, col5, col6 = st.columns(3)

col4.altair_chart(mass_chart, use_container_width=True)

col5.altair_chart(dist, use_container_width=True)

col6.altair_chart(snr, use_container_width=True)



st.markdown('### Select an event')
#create chart
event_chart = px.scatter(df, x="total_mass_source", y="commonName")

event_chart.update_layout(
    xaxis_title="Total Mass",
    yaxis_title="Event",
    yaxis=dict(
        tickmode='array',
        title='Event',  # Set your y-axis title here
    ),
    margin=dict(l=150),  # Adjust the left margin value to create more space for the y-axis labels
)
event_chart.update_xaxes(range=[0,200],
    title_font = {"size": 20},
    title_standoff = 20,
)
event_chart.update_yaxes(range=[0,20],
    title_font = {"size": 20},
    title_standoff = 50,
)

#get user input
select_event = plotly_events(event_chart, click_event=True)
selected_gwc_event = [point['y'] for point in select_event]

# Display the selected points
st.write("Selected Event:", selected_gwc_event)

# Load GPS information corresponding to the selected event
if selected_gwc_event:
    event_name = selected_gwc_event[0]
    gps_info = event_gps(event_name)
    if gps_info:
        st.write("GPS Information:", gps_info)
    else:
        st.write("GPS Information not available for the selected event.")
else:
    st.write("Select an event by clicking on the plot.")

detectorlist = ['H1', 'L1', 'V1']
detector = st.selectbox("Pick the Detector", detectorlist)
st.write('Detector: ', detector)

#print timeseries and gps info to confirm
segment = (int(gps_info)-5, int(gps_info)+5)
st.write(segment)

ldata = TimeSeries.fetch_open_data(detector, *segment, verbose=True)
st.write(ldata)

specgram = ldata.spectrogram(2, fftlength=1, overlap=.5)**(1/2.)

plot = specgram.imshow(norm='log', vmin=5e-24, vmax=1e-19)
ax = plot.gca()
ax.set_yscale('log')
ax.set_ylim(10, 2000)
ax.colorbar(
    label=r'Gravitational-wave amplitude'
)

st.pyplot(plot)


#--Add
#toggle between confirmed and marginal

#--Fix errors
#startup gps error
#detector error
#presistance issue with graph