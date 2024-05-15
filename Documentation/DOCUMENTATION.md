# Documentation
[Back to Read Me](https://github.com/brandonyee-cs/StockTrackOnline)

## Running Beta Services:
Clone the Repository

Change Directories into the Beta Directory
```
cd beta
```

Install Necessary Packages.
```
pip install -r dependancies.txt
```

For this setup you need both an [Alpha Vantage API Key](https://www.alphavantage.co/), and a [Trading Economics API Key](https://tradingeconomics.com/).

Once you have both API keys create a seperate csv file named `config.csv` containing the API name, API key.

```
alphavantageapi, [API Key]
tradingeconomicsapi, [API Key]
```

Then open config.py with your preferred text editor and replace the value of `config_path` with the path of your config file. (If config.csv is not in the directory you may need to copy the entire path).

```
config_path = 'PUT YOUR CONFIG PATH HERE'
```

Then run:

```
python3 beta.py
```