import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
from streamlit_elements import elements, mui, html
from streamlit_elements import dashboard
import plotly.graph_objects as go


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
apptitle = 'GWC Global View'

st.set_page_config(layout='wide', page_title=apptitle)

#Title the app
st.title('GWC Dashboard')

# Fetch the data from the URL and load it into a DataFrame
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv("https://gwosc.org/eventapi/csv/GWTC/")

df = get_data()

count = df.commonName.unique().size

#Top Dashboard Elements
st.markdown('### Metrics')

col1, col2, col3 = st.columns(3)

col1.metric(
    label="Total Observations to date",
    value=(count),
)

col2.metric(
    label="Pie Chart",
    value=("test"),

)

col3.metric(
    label="Total Observation time",
    value=("test")
)


#create chart
event_chart = px.scatter(df, x="total_mass_source", y="commonName")

event_chart.update_layout(
    yaxis=dict(
        tickmode='array',
        title='Event',  # Set your y-axis title here
    ),
    margin=dict(l=150),  # Adjust the left margin value to create more space for the y-axis labels
)
event_chart.update_layout(
    xaxis_title="Total Mass",
    yaxis_title="Event",
)
event_chart.update_yaxes(range=[0,30],
    title_font = {"size": 15},
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

#print timeseries and gps info to confirm
segment = (int(gps_info)-5, int(gps_info)+5)
st.write(segment)

ldata = TimeSeries.fetch_open_data('L1', *segment, verbose=True)
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

#Fix errors
#startup gps error
#set up detector selector
#detector error
#presistance issue with graph
#missing total mass 