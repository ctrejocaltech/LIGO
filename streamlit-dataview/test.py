import gwpy
from gwosc.datasets import event_gps
from gwpy.timeseries import TimeSeries
from scipy.signal import get_window

gps = event_gps('GW190412')
segment = (int(gps)-5, int(gps)+5)

ldata = TimeSeries.fetch_open_data('L1', *segment, verbose=True)

fft = ldata.fft()

window = get_window('hann', ldata.size)
lwin = ldata*window

fftamp = lwin.fft().abs()

asd = ldata.asd(fftlength=2, method="median")

plot = asd.plot()
plot.show(warn=False)