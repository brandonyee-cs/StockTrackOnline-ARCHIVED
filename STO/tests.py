import unittest
from STO.analysis import stockAnalysis
from priceprediction import LSTM
from companyprofile import CompanyProfile
from calendar import economicCalendar

class TestFeatures(unittest.TestCase):
    def setUp(self):
        # Initialize the classes with test data
        self.analysis = stockAnalysis('AAPL')
        self.lstm = LSTM('AAPL')
        self.profile = CompanyProfile('AAPL')
        self.calendar = economicCalendar()

    def test_analysis(self):
        # Test the analysis features
        self.analysis.quickedit()
        self.assertIsNotNone(self.analysis.getstock_df())

    def test_lstm(self):
        # Test the LSTM model
        self.lstm.processdata()
        self.assertIsNotNone(self.lstm.test)
        self.assertIsNotNone(self.lstm.predictions)

    def test_profile(self):
        # Test the company profile feature
        profile = self.profile.get_profile()
        self.assertIsNotNone(profile)

    def test_calendar(self):
        # Test the economic calendar feature
        events = self.calendar.get_upcoming_events()
        self.assertIsNotNone(events)

if __name__ == '__main__':
    unittest.main()

#REDO