from STO.analysis import stockAnalysis
from STO.calendar import EconomicCalendar
from STO.companyprofile import CompanyProfile
from STO.financials import financials
from STO.graphs import graphShow
from STO.news import news

class tests:
    def __init__(self, ticker) -> None:
        self.ticker = ticker

    def test_analysis(self):
        analysis = stockAnalysis(self.ticker)
        return self.assertIsNotNone(analysis.quickedit())

    def test_calendar(self):
        calendar = EconomicCalendar()
        return self.assertIsNotNone(calendar.get_upcoming_events())

    def test_companyprofile(self):
        profile = CompanyProfile(self.ticker)
        return self.assertIsNotNone(profile.get_profile())

    def test_financials(self):
        financial = financials(self.ticker)
        return self.assertIsNotNone(financial.get_financials())

    def test_LSTM(self):
        graph = graphShow(self.ticker)
        return self.assertIsNotNone(graph.LSMA())

def run_tests(ticker):
    testspassed = 0
    if tests(ticker).test_analysis() == True: print("Analysis test passed"); testspassed += 1
    else: print("Analysis test failed")
    if tests(ticker).test_calendar() == True: print("Calendar test passed"); testspassed += 1
    else: print("Calendar test failed")
    if tests(ticker).test_companyprofile() == True: print("Company profile test passed"); testspassed += 1
    else: print("Company profile test failed")
    if tests(ticker).test_financials() == True: print("Financials test passed"); testspassed += 1
    else: print("Financials test failed")
    if tests(ticker).test_LSTM() == True: print("LSMA test passed"); testspassed += 1
    else: print("LSTM test failed")
    print(f"{testspassed}/5 tests passed")
    
def main():
    input_ticker = input("Enter a ticker: ")
    run_tests(input_ticker)

main()