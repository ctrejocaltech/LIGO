import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_plotly_events import plotly_events
from streamlit_elements import elements, mui, html

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
from gwpy.timeseries import TimeSeries
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

st.set_page_config(page_title=apptitle)

#Title the app
st.title('GWC Dashboard')


#Dashboard elements

with elements("new_element"):
    mui.Typography("Hello Karen, hehe")


# Fetch the data from the URL and load it into a DataFrame
url = 'https://gwosc.org/eventapi/csv/GWTC/'
data = pd.read_csv(url)

#create chart
fig = px.scatter(data, x="total_mass_source", y="commonName")

fig.update_layout(
    yaxis=dict(
        tickmode='array',
        title='Event',  # Set your y-axis title here
    ),
    margin=dict(l=150),  # Adjust the left margin value to create more space for the y-axis labels
)
fig.update_layout(
    xaxis_title="Total Mass",
    yaxis_title="Event",
)
fig.update_yaxes(range=[0,30],
    title_font = {"size": 15},
    title_standoff = 50,
)

#get user input
select_event = plotly_events(fig, click_event=True)
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

