from stock_df import df
import yfinance as yf
import datetime
import pandas as pd 

class balancesheet(df):
    def __init__(self, ticker) -> None:
        super().__init__(ticker)

    def get_financials(self):
        self.financials = yf.Ticker(self.ticker)
        self.income_stmt = self.financials.get_income_stmt()
        self.balance_sheet = self.financials.get_balance_sheet()
        self.cashflow = self.financials.get_cashflow()
        self.quarterly_income_stmt = self.financials.get_income_stmt(freq='quarterly')
        self.quarterly_balance_sheet = self.financials.get_balance_sheet(freq='quarterly')
        self.quarterly_cashflow = self.financials.get_cashflow(freq='quarterly')

    def get_balance_sheet(self):
        quarter = ((datetime.date.today()).month - 1) // 3 + 1
        data = [self.income_stmt, self.balance_sheet, self.cashflow, self.quarterly_income_stmt, self.quarterly_balance_sheet, self.quarterly_cashflow]
        finacials_df = pd.DataFrame(data, columns=['Income Statement', 'Balanced Sheet', 'Cash Flow', 'Quarterly Income Statement', 'Quarterly Balanced Sheet', 'Quarterly Cash Flow'])
        finacials_df = finacials_df.set_index( [pd.Index([f'2024 Quater:{quarter}'])]) 
        return finacials_df