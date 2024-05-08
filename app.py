from flask import Flask, render_template, request
from sto import AI, GraphShow  # Import your classes from sto.py

app = Flask(__name__)

@app.route('/sentiment', methods=['POST'])
def sentiment():
    ticker = request.form['ticker']
    ai = AI()
    sentiment = ai.SA(ticker)
    return sentiment

@app.route('/news', methods=['POST'])
def news():
    ticker = request.form['ticker']
    ai = AI()
    news = ai.get_news(ticker)
    return news

@app.route('/plot', methods=['POST'])
def plot():
    ticker = request.form['ticker']
    graph = GraphShow()
    plot = graph.plot_close(ticker)
    return plot

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)