from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)

# Create random data with numpy
import numpy as np

archive = open('primeiro_teste.txt', 'r')
text = archive.readlines()
frequencies = []

for line in text:
    frequencies.append(int(line))
    
#print frequencies    
archive.close() 

frequencies = np.asarray(frequencies, dtype=np.intc)

def moving_average(a, n):
    wi = 0
    wf = n
    j = n
    print a.size
    print a
    while j < a.size:
        window = a[wi:wf]
        mean_window = np.mean(window)
        i = wi
        print wf
        while i < wf:
            if a[i] > mean_window:
                a[i] = int(mean_window)
            i = i + 1
        wi = wi + 1
        wf = wf + 1
        j = j + 1
    return a


filtered = moving_average(frequencies, 10)

axis_x = np.linspace(0, (filtered.size)-1, filtered.size)
axis_y = filtered

# Create a trace
trace = go.Scatter(
    x = axis_x,
    y = axis_y
)

data = [trace]

iplot(data, filename='basic-line')