from plotly.offline import plot, iplot
import plotly.graph_objs as go


def plot_df(df):
    traces = [go.Scatter(x=df.index, y=df[c], name=c) for c in df.columns]
    fig = go.Figure(data=traces)
    plot(traces)
    return fig


def iplot_df(df, cols=None):
    if cols is None:
        traces = [go.Scatter(x=df.index, y=df[c], name=c) for c in df.columns]
    else:
        traces = [go.Scatter(x=df.index, y=df[c], name=c) for c in cols]
    iplot(traces)

