import plotly.graph_objs as go
import pandas as pd
from datetime import datetime as dt
from invest.security import Security
import invest.dateutils as dateutils


class Plot:
    plotmap = []

    def __init__(self):
        self.plotmap = {a: getattr(self, a) for a in dir(self) if not a.startswith('_') and callable(getattr(self, a))}

    def plot(self, plot_name, **kwargs):
        if f'plot_{plot_name}' in self.plotmap:
            retval = getattr(self, f'plot_{plot_name}')(**kwargs)
            return retval
        else:
            raise LookupError(f"No plot exists with name {plot_name}")

    @staticmethod
    def plot_rsi(security: Security, rsi_period=20, date_range=None):
        temp_df = security.data.copy()
        temp_df = pd.merge(security.get_rsi(rsi_period), temp_df, on='date')

        if not date_range:
            date_range = [dt.strftime(min(temp_df['date']), '%Y-%m-%d'), dt.strftime(dt.now(), '%Y-%m-%d')]
        date_range[0] = min(temp_df['date']) if not date_range[0] else date_range[0]
        date_range[1] = dt.strftime(dt.now(), '%Y-%m-%d') if not date_range[1] else date_range[1]
        temp_df = temp_df.loc[(temp_df['date'] > date_range[0]) & (temp_df['date'] <= date_range[1])]

        data = [go.Scatter(
                    x=temp_df['date'],
                    y=temp_df[f'rsi_{rsi_period}dy'] / 100,
                    line=dict(width=1),
                    name='RSI',
                    yaxis='y2'
                )]
        for hline in [.3, .7]:
            data.append(
                go.Scatter(
                    x=[dt.strptime(date_range[0], '%Y-%m-%d'), dt.strptime(date_range[1], '%Y-%m-%d')],
                    y=[hline, hline],
                    line=dict(width=1),
                    line_color='rgba(196,196,196,1)',
                    showlegend=False,
                    yaxis='y2'
                )
            )

        fig = go.Figure(data=data)

        fig['layout'] = dict(title=f'{security.symbol} RSI Analysis')
        fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
        fig['layout']['xaxis'] = dict(rangeselector=dict(visible=True),
                                      range=[dt.strptime(date_range[0], '%Y-%m-%d'),
                                             dt.strptime(date_range[1], '%Y-%m-%d')])
        fig['layout']['yaxis'] = dict(showticklabels=False, title="Value")
        fig['layout']['legend'] = dict(orientation='h', y=.99, x=0.65, yanchor='bottom')
        fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)

        rangeselector = dict(
            visible=True,
            x=0.01, y=1,
            bgcolor='rgba(150, 200, 250, 0.4)',
            font=dict(size=13),
            buttons=list([
                dict(count=1, label='reset', step='all'),
                dict(count=5, label='5yr', step='year', stepmode='backward'),
                dict(count=3, label='3yr', step='year', stepmode='backward'),
                dict(count=1, label='1yr', step='year', stepmode='backward'),
                dict(count=3, label='3mo', step='month', stepmode='backward'),
                dict(count=1, label='1mo', step='month', stepmode='backward'),
                dict(step='all')
            ]))
        fig['layout']['xaxis']['rangeselector'] = rangeselector
        return fig

    @staticmethod
    def plot_general(security: Security, bollinger_period=20, bollinger_stddev_factor=1, date_range=None):
        temp_df = security.data.copy()
        temp_df = pd.merge(security.get_bollinger_bands(period=bollinger_period, stddev_factor=bollinger_stddev_factor),
                           temp_df, on='date')
        temp_df = pd.merge(temp_df, security.get_sma_price(period=bollinger_period), on='date')

        if not date_range:
            date_range = [dt.strftime(min(temp_df['date']), '%Y-%m-%d'), dt.strftime(dt.now(), '%Y-%m-%d')]
        date_range[0] = min(temp_df['date']) if not date_range[0] else date_range[0]
        date_range[1] = dt.strftime(dt.now(), '%Y-%m-%d') if not date_range[1] else date_range[1]
        temp_df = temp_df.loc[(temp_df['date'] > date_range[0]) & (temp_df['date'] <= date_range[1])]

        fig = go.Figure(
            data=[go.Candlestick(
                x=temp_df['date'],
                open=temp_df['open'],
                high=temp_df['high'],
                low=temp_df['low'],
                close=temp_df['close'],
                name="Candlestick",
                yaxis='y2'
            ),
                go.Bar(
                    x=temp_df['date'],
                    y=temp_df['volume'],
                    marker_color='rgba(128,128,255,1)',
                    name='VLM'
                ),
                go.Scatter(
                    x=temp_df['date'],
                    y=temp_df[f'bollinger_upper_{bollinger_period}dy_{bollinger_stddev_factor}std'],
                    line_color='rgba(196,196,196,.7)',
                    line=dict(width=1),
                    showlegend=False,
                    name='Upper Bollinger Band',
                    yaxis='y2'
                ),
                go.Scatter(
                    x=temp_df['date'],
                    y=temp_df[f'bollinger_lower_{bollinger_period}dy_{bollinger_stddev_factor}std'],
                    line_color='rgba(196,196,196,.7)',
                    line=dict(width=1),
                    showlegend=False,
                    name='Lower Bollinger Band',
                    yaxis='y2'
                ),
                go.Scatter(
                    x=temp_df['date'],
                    y=temp_df[f'sma_price_{bollinger_period}dy'],
                    line_color='rgba(64,64,64,.7)',
                    line=dict(width=1),
                    name='EMA',
                    yaxis='y2'
                )
            ])

        fig['layout'] = dict(title=f'Analysis for {security.symbol}')
        fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
        fig['layout']['xaxis'] = dict(rangeselector=dict(visible=True))
        fig['layout']['yaxis'] = dict(domain=[0, 0.2], showticklabels=False, title="Volume")
        fig['layout']['yaxis2'] = dict(domain=[0.2, 0.95], showticklabels=True, title='USD')
        fig['layout']['legend'] = dict(orientation='h', y=.99, x=0.65, yanchor='bottom')
        fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)

        rangeselector = dict(
            visible=True,
            x=0.01, y=1,
            bgcolor='rgba(150, 200, 250, 0.4)',
            font=dict(size=13),
            buttons=list([
                dict(count=1, label='reset', step='all'),
                dict(count=5, label='5yr', step='year', stepmode='backward'),
                dict(count=3, label='3yr', step='year', stepmode='backward'),
                dict(count=1, label='1yr', step='year', stepmode='backward'),
                dict(count=3, label='3mo', step='month', stepmode='backward'),
                dict(count=1, label='1mo', step='month', stepmode='backward'),
                dict(step='all')
            ]))
        fig['layout']['xaxis']['rangeselector'] = rangeselector
        return fig
