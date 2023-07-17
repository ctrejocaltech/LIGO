import gwpy
from gwosc.datasets import event_gps
from scipy.signal import get_window 


gps = event_gps('GW190412')

segment = (int(gps)-5, int(gps)+5)


from gwpy.timeseries import TimeSeries
ldata = TimeSeries.fetch_open_data('L1', *segment, verbose=True)

window = get_window('hann', ldata.size)
lwin = ldata*window 

fftamp = lwin.fft().abs()
plot = fftamp.plot(xscale="log", yscale="log") 
plot.show(warn=False) 

