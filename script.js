document.getElementById('tickerForm').addEventListener('submit', function(event) {
    event.preventDefault();
    var ticker = document.getElementById('ticker').value;
    fetch('/sentiment', {
        method: 'POST',
        body: JSON.stringify({ticker: ticker}),
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('sentiment').textContent = data;
    });
    fetch('/news', {
        method: 'POST',
        body: JSON.stringify({ticker: ticker}),
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('news').textContent = data;
    });
    fetch('/plot', {
        method: 'POST',
        body: JSON.stringify({ticker: ticker}),
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('plot').textContent = data;
    });
});