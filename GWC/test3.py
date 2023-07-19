import pandas as pd
import altair as alt
import numpy as np
import streamlit as st


# Fetch the data from the URL and load it into a DataFrame
url = 'https://gwosc.org/eventapi/csv/GWTC/'
data = pd.read_csv(url)



# Create the Altair histogram
hist = alt.Chart(data, title='Total mass by Event').mark_circle(size=60).encode(
    alt.X('total_mass_source', title='Mass'),
    alt.Y('commonName:N', title='Event'),
    tooltip = ['total_mass_source', 'commonName', 'GPS', 'final_mass_source'], #check why this scales the y-axis down
    ).add_params(
    ).interactive()


# Show the chart
st.altair_chart(hist)

