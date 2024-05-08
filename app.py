from flask import Flask, render_template, request
from sto import df, stockAnalysis, publicSentiment, LSTM, graphShow # Import your classes from sto.py

app = Flask(__name__)

@app.route('/publicsentiment', methods=['POST'] )
def publicsentiment():
    ticker = request.form['ticker']
    ps = df(ticker); ps = publicSentiment()
    sentiment = ps.SA(); news = ps.get_news()
    return sentiment, news

@app.route('/stockanalysis', methods=['POST'])
def stockanalysis():
    ticker = request.form['ticker']
    sa = df(ticker) 
    sa = stockAnalysis(df)
    stock_df = sa.EMA(); stock_df = sa.SMA(); stock_df = sa.RSI(); stock_df = sa.BOLLINGER_BANDS()
    return stock_df

@app.route('/lstmplot', methods=['POST'])
def lstmgraphshow():
    ticker = request.form['ticker']
    graph = df(ticker); graph = LSTM()
    graph.forward(); graph.processdata()
    graph = graphShow(); plot = graph.plot_predictions()
    return plot

@app.route('/plot', methods=['POST'])
def standaredgraphshow():
    ticker = request.form['ticker']
    graph = df(ticker); graph = graphShow()
    plot = graph.plot_close()
    return plot

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)