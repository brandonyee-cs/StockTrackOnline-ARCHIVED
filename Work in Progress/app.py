from flask import Flask, render_template, request
from STO.calendar import EconomicCalendar
from STO.analysis import stockAnalysis
from STO.graphs import graphShow
from STO.news import news
from STO.priceprediction import LSTM

app = Flask(__name__)

#Stock Analysis
@app.route('/getstockanalysis', methods=['POST'])
def stockanalysis():
    ticker = request.form['ticker']
    sa = stockAnalysis(ticker)
    sa.add_conventional_indicators(); sa.quickedit()
    stock_df = sa.getstock_df()
    return stock_df

#First Interactive Graph
@app.route('/plot', methods=['POST'])
def standaredgraphshow():
    ticker = request.form['ticker']
    graph = graphShow()
    plot = graph.plot_close()
    return plot

#Stock Screener
@app.route('/getstockscreener', methods=['POST'])
def stockscreener():
    ticker = request.form['ticker']
    sa = stockAnalysis(ticker)
    info = sa.stock_screener()
    return info

#Sentiment Analysis
@app.route('/getpublicsentiment', methods=['POST'] )
def publicsentiment():
    ticker = request.form['ticker']
    ps = news(ticker)
    sentiment = ps.SA()
    return sentiment

#News
@app.route('/getnews', methods=['POST'])
def getnews():
    ticker = request.form['ticker']
    ps = news(ticker)
    news = ps.get_news()
    return news

#LSTM Prediction
@app.route('/lstmplot', methods=['POST'])
def lstmgraphshow():
    ticker = request.form['ticker']
    graph = graphShow(ticker)
    plot = graph.plot_predictions()
    return plot

@app.route('/getcalendar', methods=['POST'])
def getcalendar():
    ec = EconomicCalendar()
    events = ec.get_upcoming_events()
    return events

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)