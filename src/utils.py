from plotly.offline import plot, iplot
import plotly.graph_objs as go


def plot_df(df):
    traces = [go.Scatter(x=df.index, y=df[c], name=c) for c in df.columns]
    plot(traces)


def iplot_df(df):
    traces = [go.Scatter(x=df.index, y=df[c], name=c) for c in df.columns]
    iplot(traces)

