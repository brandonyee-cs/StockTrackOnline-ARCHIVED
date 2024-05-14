from STO.analysis import stockAnalysis
from STO.calendar import EconomicCalendar
from STO.companyprofile import CompanyProfile
from STO.financials import financials
from STO.graphs import graphShow
from STO.priceprediction import LSTM
from STO.news import news

class main():
    while True:
        print('1. Analysis')
        print('2. Calendar')
        print('3. Company Profile')
        print('4. Financials')
        print('5. Graphs')
        print('6. Price Prediction')
        print('7. News')
        print('8. Exit')
        choice = int(input('Enter your choice: '))
        if choice == 1:
            ticker = input('Enter the ticker: ')
            analysis = stockAnalysis(ticker)
            print(analysis.quickedit())
        elif choice == 2:
            calendar = EconomicCalendar()
            events = calendar.get_upcoming_events()
            print(events)
        elif choice == 3:
            ticker = input('Enter the ticker: ')
            profile = CompanyProfile(ticker)
            print(profile.get_profile())
        elif choice == 4:
            ticker = input('Enter the ticker: ')
            financial = financials(ticker)
            financial.get_financials()
            print(financial.get_balance_sheet())
        elif choice == 5:
            ticker = input('Enter the ticker: ')
            graph = graphShow(ticker)
            graph.show_graph()
        elif choice == 6:
            ticker = input('Enter the ticker: ')
            lstm = graphShow(ticker)
            lstm.plot_predictions()
        elif choice == 7:
            ticker = input('Enter the ticker: ')
            news = news(ticker)
            news.get_news()
        elif choice == 8:
            break
        else:
            print('Invalid choice')

main()