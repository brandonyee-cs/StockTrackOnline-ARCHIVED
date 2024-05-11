from flask import Flask, render_template, request
from sto import df, publicSentiment, LSTM, graphShow, stockAnalysis # Import your classes from sto.py

app = Flask(__name__)

@app.route('/getstockanalysis', methods=['POST'])
def stockanalysis():
    ticker = request.form['ticker']
    sa = stockAnalysis(ticker)
    sa.add_conventional_indicators(); sa.quickedit()
    stock_df = sa.getstock_df()
    return stock_df

@app.route('/plot', methods=['POST'])
def standaredgraphshow():
    ticker = request.form['ticker']
    graph = df(ticker); graph = graphShow()
    plot = graph.plot_close()
    return plot

@app.route('/getstockscreener', methods=['POST'])
def stockscreener():
    ticker = request.form['ticker']
    sa = stockAnalysis(ticker)
    info = sa.stock_screener()
    return info


@app.route('/getpublicsentiment', methods=['POST'] )
def publicsentiment():
    ticker = request.form['ticker']
    ps = publicSentiment()
    sentiment = ps.SA()
    return sentiment

@app.route('/getnews', methods=['POST'])
def news():
    ticker = request.form['ticker']
    ps = publicSentiment()
    news = ps.get_news()
    return news


@app.route('/lstmplot', methods=['POST'])
def lstmgraphshow():
    ticker = request.form['ticker']
    graph = df(ticker); graph = LSTM()
    graph.forward(); graph.processdata()
    graph = graphShow(); plot = graph.plot_predictions()
    return plot

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)