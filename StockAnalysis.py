import pandas as pd
import matplotlib.pyplot as plt, mpld3
from matplotlib.ticker import MaxNLocator
import numpy as np

import datetime
import time
from threading import Timer
import StringIO
import requests
#from mpldatacursor import datacursor
from matplotlib.finance import candlestick_ohlc
from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override() # <== that's all it takes :-)
import pandas as pd


import sqlite3
import pandas as pd

class DataBase(object):
    def __init__(self, StockList=['^IXIC', 'SPY'], DBName='StockPrice.db'):
        StockList = ['^IXIC', 'NDAQ', 'SPY', 'JNPR', 'WMT', 'MAT', 'ZG', 'AMAT', 'EBAY', 'GOLD', 'VMW', 'NVDA', 'TSLA']
        GenlStockList = ['NDAQ', 'SPY', 'GOLD']

        SemiStockList = ['MU', 'NVDA', 'AMD', 'INTC', 'AMAT', 'ASML', 'KLAC', 'COHR', 'QCOM', 'VIAV', 'IIVI']
        EcomStockList = ['BABA', 'AMZN', 'EBAY', 'PYPL']
        CloudStockList = ['GOOG', 'GOOGL', 'VMW', 'JNPR', 'TWTR', 'MSFT', 'NOW', 'SPLK', 'SAP', 'PYPL', 'IBM', 'NOW',
                          'BABA']
        CarStockList = ['TSLA', 'F', 'GM', 'FCAU']
        BioMedStockList = ['JNJ', 'ILMN', 'WBA']
        LaserStockList = ['OCLR', 'IPGP', 'COHR']
        AIStockList = []
        NewStockList = ['CLDR', 'SNAP']

        GoodStockList1 = ['NDAQ', 'SPY', 'GOLD', 'VMW', 'NVDA', 'AMD', 'COHR', 'NOW', 'PYPL', 'BABA']
        GoodStockList2 = ['JNPR', 'ASML', 'LRCX', 'AMAT', 'TSLA', 'AAPL']
        GoodStockList3 = ['^IXIC', 'FCAU']
        GoodStockList4 = ['MU', 'IPGP']

        FinanceStockList = ['BAC', 'GS']

        AllStockList = GenlStockList + SemiStockList + StockList + GenlStockList + EcomStockList \
                       + CloudStockList + CarStockList + BioMedStockList + \
                       LaserStockList + SemiStockList

        self.StockList = list(set(AllStockList))
        self.DBName = DBName


    def loadTableIntoSQLite3(self, df, TableName='trx', DataBaseName='test.db'):
        """create a new table by loading from a dataframe"""
        conn = sqlite3.connect(DataBaseName)
        df.to_sql(TableName, conn, if_exists='fail', index=False)
        conn.close()
        print TableName + 'is loaded.'

    def dropTableinSQLite3(self, TableName, DataBaseName='test.db'):
        conn = sqlite3.connect(DataBaseName)
        c = conn.cursor()
        c.execute('drop table ' + TableName + ';')
        conn.close()

    def updateTableInSQLite3(self, df, TableName='trx', DataBaseName='test.db'):
        """update an existing table by loading from a dataframe"""
        conn = sqlite3.connect(DataBaseName)
        c = conn.cursor()
        KeyName = 'Date'
        query = ' '.join(['select', KeyName, 'from', '\''+TableName+'\'', ';'])
        print query
        keys = [key[0] for key in set(c.execute(query))]
        if bool(set(df[KeyName]).intersection(set(keys))):
            print 'adding records'
            df.to_sql(TableName, conn, if_exists='append', index=False)
        else:
            print 'found overlapping, new data is not loaded.'
        conn.close()

    def loadDataFromSQLite3(self, TableName='trx', DataBaseName='test.db'):
        """load data from DataBase"""
        conn = sqlite3.connect(DataBaseName)
        c = conn.cursor()
        query = ' '.join(['select * from ', '\''+TableName+'\'', ';'])
        print query
        df = pd.read_sql(query, conn, index_col='Date')
        conn.close()
        #print df
        return df

    def replaceTableInSQLite3(self, df, TableName='trx', DataBaseName='test.db'):
        dropTableinSQLite3(TableName=TableName, DataBaseName=DataBaseName)
        loadTableIntoSQLite3(df, TableName=TableName, DataBaseName=DataBaseName)


    def DownloadStocks(self):
        [year, month, day] = datetime.date.today().isoformat().split('-')
        today = [str(int(month)), str(int(day)), year]
        print today
        for stockname in self.StockList:
            print stockname
            S = Stock(stockname=stockname, eDate=today, DataSource='Yahoo')
            self.loadTableIntoSQLite3(S.StockData, TableName=stockname, DataBaseName=self.DBName)

    def UpdateStocks(self):
        [year, month, day] = datetime.date.today().isoformat().split('-')
        today = [str(int(month)), str(int(day)), year]
        print today
        for stockname in self.StockList:
            print stockname
            S = Stock(stockname=stockname, sDate=today, eDate=today, DataSource='Yahoo')
            print S.StockData
            self.updateTableInSQLite3(S.StockData, TableName=stockname, DataBaseName=self.DBName)


            # loadTableIntoSQLite3(df = df)
    # dropTableinSQLite3(TableName=' trx', DataBaseName = 'test.db')
    # updateTableInSQLite3(df.iloc[0:5])
    # print loadDataFromSQLite3()



class TimeConverter(object):

    Month = {'1': 'Jan', '2': 'Feb', '3': 'Mar', '4': 'Apr', '5': 'May',
             '6': 'Jun', '7': 'Jul', '8': 'Aug', '9': 'Sep', '10': 'Oct',
             '11': 'Nov', '12': 'Dec'}
    def __init__(self):
        pass

    def dateNum2Str(self, date=['8', '2', '2012']):
        """
        Example:
        :param date:[8, 2, 2012]
        :return: 'Aug+2%2C+2012'
        """
        return self.Month[date[0]]+'+'+str(date[1])+'%2C+'+str(date[2])


class Stock(object):

    # def __init__(self, stockname = 'NVDA', sDate=['3', '12', '2000'], eDate=['1', '27', '2017']):
    #     self.YahooFinanceURL = 'http://real-chart.finance.yahoo.com/table.csv?s='
    #     self.stockName = stockname
    #     self.startDate = sDate
    #     self.endDate = eDate
    #     self.StockData = self.downloadHistoryCSVviaURL(self.composeYahooFinanceQueryURL())

    #https://query1.finance.yahoo.com/v7/finance/download/CSV?period1=1502161290&period2=1504839690&interval=1d&events=history&crumb=6hysORdGG5z
    #https://query1.finance.yahoo.com/v7/finance/download/COHR?period1=1502161636&period2=1504840036&interval=1d&events=history&crumb=6hysORdGG5z
    #javascript:getQuotes(true);
    def __init__(self, stockname = 'NVDA', sDate=['3', '12', '2010'], eDate=['8', '30', '2017'], DataSource='Google'):
        self.YahooFinanceURL = 'http://real-chart.finance.yahoo.com/table.csv?s='
        self.YahooFinanceURLBase = 'https://query1.finance.yahoo.com/v7/finance/download/'
        self.stockName = stockname
        self.startDate = sDate
        self.endDate = eDate

        #print "http://www.google.com/finance/historical?q=NASDAQ:ADBE&startdate=Mar+12%2C+2002&enddate=Aug+27%2C+2017&output=csv"
        if DataSource == 'Google':
            self.StockData = self.downloadHistoryCSVviaURLFromGoogle(self.composeGoogleFinanceQueryURL_Version2())
        elif DataSource == 'Yahoo':
            self.StockData = self.downloadHistoryCSVviaYahooFinancePackage()
            self.StockData = self.StockData.convert_objects(convert_numeric=True)
        elif DataSource == 'DataBase':
            self.StockData = DataBase().loadDataFromSQLite3(TableName=stockname, DataBaseName='StockPrice.db')

        elif StockData is None:
            while True:
                try:
                    self.StockData = self.downloadHistoryCSVviaYahooFinancePackage()
                    self.StockData = self.StockData.convert_objects(convert_numeric=True)
                    if self.StockData is None or 'Close' not in self.StockData:
                        raise KeyError('incomplete data downloading')


                except KeyError:
                    continue
                break



        newQuery = """
        https://query1.finance.yahoo.com/v7/finance/download/JNPR?period1=1493340069&period2=1495932069&interval=1d&events=history&crumb=H0zm4TDlHz1
        """
    def downloadHistoryCSVviaYahooFinancePackage(self):
        name = self.stockName
        sD = [int(x) for x in self.startDate]
        eD = [int(x) for x in self.endDate]
        sDD = datetime.datetime(sD[2], sD[0], sD[1]).strftime('%Y-%m-%d')
        eDD = datetime.datetime(eD[2], eD[0], eD[1]+1).strftime('%Y-%m-%d')

        StockData = pdr.get_data_yahoo(name, start=sDD, end=eDD)
        #print self.StockData
        StockData['Date'] = StockData.index
        StockDataFrame = StockData.iloc[::-1]


        return StockDataFrame

    
    def downloadHistoryCSVviaURL(self, url):
        print url
        StockDataFrame = pd.read_csv(url)
        print 'trouble if you dont see this line'
        # the data resolution is one day that is
        # a big restriction to analyze stock
        # fluctuation details
        self.StockData = StockDataFrame
        return StockDataFrame

    def downloadHistoryCSVviaURLFromGoogle(self, url):
        print url
        StockDataFrame = pd.read_csv(url)

        #print StockDataFrame

        StockDataFrame = pd.DataFrame(StockDataFrame.values, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        StockDataFrame['Date'] = pd.to_datetime(StockDataFrame['Date'])
        StockDataFrame['Date'] = StockDataFrame['Date'].dt.strftime('%Y-%m-%d')
        StockDataFrame=pd.DataFrame(StockDataFrame.values,index=StockDataFrame['Date'],columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

        StockDataFrame.to_csv('test.csv')
        if StockDataFrame.isin(['-']).values.any():
            StockDataFrame = None
            print 'google has missing data for ' + self.stockName + ', so try yahoo.'

        self.StockData = StockDataFrame
        return StockDataFrame

    def downloadHistoryCSVviaURLFromYahoo(self, url):
        print url
        #s = requests.get(url).content
        #StockDataFrame = pd.read_csv(StringIO.StringIO(s.decode('utf-8')))
        print 'does the url looks good?'
        StockDataFrame = pd.read_csv(url)
        print StockDataFrame
        print 'trouble if you dont see this line'
        # the data resolution is one day that is
        # a big restriction to analyze stock
        # fluctuation details
        self.StockData = StockDataFrame
        return StockDataFrame


    def composeYahooFinanceQueryURL(self):
        #YahooFinanceURL = 'http://real-chart.finance.yahoo.com/table.csv?s='
        name = self.stockName
        sD = self.startDate
        eD = self.endDate
        PeriodString = '&d=' + eD[0] + '&e=' + eD[1] + '&f=' + eD[2] +\
        '&g=d&a=' + sD[0] + '&b=' + sD[1] + '&c=' + sD[2] + '&ignore=.csv'
        queryString = self.YahooFinanceURL + name + PeriodString
        return queryString

    def composeYahooFinanceQueryURL_Version2(self):
        # https://query1.finance.yahoo.com/v7/finance/download/CSV?period1=1502161290&period2=1504839690&interval=1d&events=history&crumb=6hysORdGG5z
        # https://query1.finance.yahoo.com/v7/finance/download/COHR?period1=1502161636&period2=1504840036&interval=1d&events=history&crumb=6hysORdGG5z

        # YahooFinanceURL = 'http://real-chart.finance.yahoo.com/table.csv?s='
        name = self.stockName
        sD = self.startDate
        eD = self.endDate
        PeriodString = 'period1=1493340069&period2=1495932069&interval=1d&events=history&crumb=H0zm4TDlHz1'
        queryString = self.YahooFinanceURLBase + name + '?' + PeriodString

        return queryString

    def composeYahooFinanceQueryURL_Version3(self):
        # https://query1.finance.yahoo.com/v7/finance/download/CSV?period1=1502161290&period2=1504839690&interval=1d&events=history&crumb=6hysORdGG5z
        # https://query1.finance.yahoo.com/v7/finance/download/COHR?period1=1502161636&period2=1504840036&interval=1d&events=history&crumb=6hysORdGG5z
        self.YahooFinanceURLBase = 'https://query1.finance.yahoo.com/v7/finance/download/'
        name = self.stockName
        sD = self.startDate
        eD = self.endDate
        sUnixTime = int(time.mktime(datetime.datetime(int(sD[2]), int(sD[0]), int(sD[1]), 0, 0).timetuple()))
        eUnixTime = int(time.mktime(datetime.datetime(int(eD[2]), int(eD[0]), int(eD[1]), 23, 0).timetuple()))
        PeriodString = 'period1=' + str(sUnixTime) + '&period2=' + str(eUnixTime)+ '&interval=1d&events=history&crumb=6hysORdGG5z'
        queryString = self.YahooFinanceURLBase + name + '?' + PeriodString

        return queryString


    def composeGoogleFinanceQueryURL_Version1(self):

        """
            url example:
            http://www.google.com/finance/historical?q=NASDAQ:ADBE&startdate=Jan+01%2C+2009&enddate=Aug+2%2C+2012&output=csv

            :return:
            """

        GoogleFinanceURL = 'http://www.google.com/finance/'
        # StockName = 'NASDAQ'
        # StartDate = 'Jan+01%2C+2009'
        # EndDate = 'Aug+2%2C+2012'
        StockName = self.stockName
        StartDate = TimeConverter().dateNum2Str(self.startDate)
        EndDate = TimeConverter().dateNum2Str(self.endDate)
        QueryForHistoricalData = 'historical?q={}:ADBE&startdate={}&enddate={}&output=csv'.format(StockName, StartDate,
                                                                                                  EndDate)
        Query = GoogleFinanceURL + QueryForHistoricalData

        return Query

    def composeGoogleFinanceQueryURL_Version2(self):

        """
            url example:
            http://www.google.com/finance/historical?q=NASDAQ:ADBE&startdate=Jan+01%2C+2009&enddate=Aug+2%2C+2012&output=csv

            :return:
            """

        GoogleFinanceURL = 'http://www.google.com/finance/'
        # StockName = 'NASDAQ'
        # StartDate = 'Jan+01%2C+2009'
        # EndDate = 'Aug+2%2C+2012'
        StockName = self.stockName
        StartDate = TimeConverter().dateNum2Str(self.startDate)
        EndDate = TimeConverter().dateNum2Str(self.endDate)
        QueryForHistoricalData = 'historical?q={}&startdate={}&enddate={}&output=csv'.format(StockName, StartDate,
                                                                                                  EndDate)
        Query = GoogleFinanceURL + QueryForHistoricalData

        return Query

    def UnitTest01(self):
        #url = 'http://real-chart.finance.yahoo.com/table.csv?s=YHOO&d=4&e=13&f=2016&g=d&a=3&b=12&c=1996&ignore=.csv'
        url = self.composeYahooFinanceQueryURL()
        
        print self.downloadHistoryCSVviaURL(url)

    def addDerivatedData(self):
        pass

    def addMidOpenClose(self):
        #print 'addMidOpenClose starts'
        #print type(self.StockData['Open'][1])
        self.StockData['MidOpenClose'] = \
            (self.StockData['Open']+self.StockData['Close']) / 2.0
        #print 'addMidOpenClose works'

    def addCloseSubOpen(self):
        self.StockData['Close-Open'] = \
            (self.StockData['Close'] - self.StockData['Open'])

    def addDailyVariationPercentage(self):
        """add percentage of daily variation relative to the
        open price: (close - open)/open * 100
        """
        self.StockData['DayVar_Open%'] = \
            (self.StockData['Close'] - self.StockData['Open']) \
            / self.StockData['Open'] * 100

    def addMidHighLow(self):
        self.StockData['MidHighLow'] = \
            (self.StockData['High']+self.StockData['Low']) / 2.0

    def addHighSubLow(self):
        self.StockData['High-Low'] = \
            (self.StockData['High'] - self.StockData['Low'])

    def addMaxDailyVariationPercentage(self):
        """add percentage of maximum daily variation
        relative to the open price: (high - low)/open * 100
        """
        self.StockData['MaxDayVar_Open%'] = \
            (self.StockData['High'] - self.StockData['Low']) \
            / self.StockData['Open'] * 100

    def add2ndDayOpenSub1stClose(self):

        ClosePrice = np.array(self.StockData['Close']\
            .truncate(before=1, after=None, axis=None, copy=True)\
            .tolist() + [0.0])
        self.StockData['2ndOpen-1stClose'] = \
            (self.StockData['Open'] - ClosePrice)

    def addOpeningRisePercentage(self):
        """
        Opening Rise = (2nd day Open - 1st day close) /
                        2nd day Open * 100
        :return:
        """
        ClosePrice = np.array(self.StockData['Close']\
            .truncate(before=1, after=None, axis=None, copy=True)\
            .tolist() + [0.0])
        self.StockData['OpeningRise%'] = \
            (self.StockData['Open'] - ClosePrice) / \
            self.StockData['Open'] * 100


    def addSecondDayOpenSubClose(self):
        self.StockData['2ndOpen-1stClose'] = \
            (self.StockData['Open'] - self.StockData['Close']) \
            / self.StockData['Close']


    def showDataList(self):
        #print type(self.StockData)
        print self.StockData
        print '-------------'

        
    def showGraph(self):
        self.StockData.plot(x='Date', y=['OpeningRise%'], subplots=False)

        ##show mid value between open and close
        ##show High, Mid and Low as a band

        ##show 1 day trends
        ##show 2 day trends
        ##show 3 day trends

        ##show 1 week trends
        ##show recent 2 week trends
        ##show recent 1 month trends
        ##show recent 2 month trends
        ##show recent 3 month trends
        ##show recent 6 month trends
        ##show recent 52 week (1 year) trends
        ##show recent 2 year trends


        plt.gca().invert_xaxis()
        plt.show()


        
    def averageGivenPeriod(self, numDays=50):
        self.StockData['Mid'] = self.StockData['Close'] #is this correct?
        #self.StockData['Mid'] = self.StockData['Open']
        #print self.StockData
        #print range(len(self.StockData['Mid']))
        average = [np.mean(self.StockData['Mid'][index:(index+numDays)]) for index in range(len(self.StockData['Mid'])-numDays)]
        #    print index
            #average[index] = 
        #    print np.mean(self.StockData['Mid'][index:(index+numDays)])
            
        #print average
        #plt.plot(average[0:2500])
        #plt.show()
        #averageOverNumDays

        standarddeviation =[np.std(self.StockData['Mid'][index:(index + numDays)]) for index in
                   range(len(self.StockData['Mid']) - numDays)]
        return (average, standarddeviation)

    def StockPriceVariation(self):
        """

        :return:
        """
        #TodayOpen, TodayClose, TodayHigh, TodayLow, MA20, StdMA20, MA5, NASQ, DJI,
    def addIntIndex(self, name='Index'):
        self.StockData['Index'] = range(0, -self.StockData.shape[0], -1)

    def addDaysBack(self, name='DaysBack'):
        self.StockData[name] = range(0, -self.StockData.shape[0], -1)

    def addGainLossLabel(self, name='NextDayGain|Loss%', days=2):
        self.StockData[name] = self.StockData['Close'].rolling(days, center=False).apply(lambda y: 100*(y[0]-y[1])/y[1])


    def addRollingAverage(self, name='NewRollingAve/5days', days=5):
        self.StockData[name]=self.StockData['Close'][::-1].rolling(days, center=False).mean()

    def addRollingStd(self, name='NewRollingStd/20days', days=20, sign=1):
        self.StockData[name] = self.StockData['Close'][::-1].rolling(days, center=False).std() * sign

    def addRollingNormalizedValue(self, name='NormalizedBollinger', days=20, sign=2):
        self.StockData[name] = (self.StockData['Close'][::-1] \
                                - self.StockData['Close'][::-1].rolling(days, center=False).mean()) \
                                / (self.StockData['Close'][::-1].rolling(days, center=False).std() * sign)

    def addRollingLstSq(self, name='k', days=20):
        retParaDic = {'k':0, 'bias':1}
        def LstSq(y):
            x = np.linspace(-days + 1, 0, days)
            A = np.vstack([x, np.ones(len(x))]).T
            #print "---" * 5
            # print np.linalg.lstsq(A, y)
            # print "---" * 5
            ans = np.linalg.lstsq(A, y[::-1])
            #k, bias = ans[0]
            #residues = ans[1][0]
            # print residues
            # print '====='
            # print k
            return ans[0][retParaDic[name]]

        self.StockData['LstSq'+name+'_'+str(days)+'Days'] = self.StockData['Close'].rolling(days, center=False).apply(lambda x: LstSq(x))


    def addRollingRSI(self, name='14-Day RSI', days=15):

        def rsi(y):
            # print
            dy = np.diff(y)
            # print dy
            # print dy[dy>0]
            # print dy[dy<0]
            # print -sum(dy[dy<0])/14.0
            # print sum(dy[dy>0])/14.0
            rs = -sum(dy[dy<=0])/sum(dy[dy>0])
            # print rs
            # print 100-100/(1+rs)
            return 100-100/(1+rs)

        #y = np.array([48.08,47.61,47.57,48.2,49.23,49.25,47.54,47.69,46.83,46.03,46.08,46.23,46.5,46.26,45.15])
        #rsi(y)


        self.StockData[name] = self.StockData['Close'].rolling(days, center=False).apply(rsi)

        self.StockData[name] = self.StockData[name].shift(-days+1)


    def addRollingStcOsc(self, name='Stochastic Osc', days=15, D=3):
        self.StockData['MaxHigh'] = self.StockData['High'].rolling(days, center=False).max().shift(-days + 1)
        self.StockData['MinLow'] = self.StockData['Low'].rolling(days, center=False).min().shift(-days + 1)
        self.StockData['StcOsc K%'] = (self.StockData['Close']-self.StockData['MinLow']) / (self.StockData['MaxHigh']-self.StockData['MinLow'])
        self.StockData['StcOsc D%'] = self.StockData['StcOsc K%'].rolling(D, center=False).mean().shift(-D + 1)
        self.StockData['StcOsc D%D%'] = self.StockData['StcOsc D%'].rolling(D, center=False).mean().shift(-D + 1)
        self.StockData.drop(['MaxHigh', 'MinLow'], axis=1, inplace=True)
        self.StockData['StcOsc K%-D%'] = self.StockData['StcOsc K%'] - self.StockData['StcOsc D%']
        self.StockData['StcOsc Diff(K%-D%)'] = self.StockData['StcOsc K%-D%'].rolling(2, center=False).apply(lambda y: y[0]-y[1])

    def loadDataFromPandas(self):
        """
        using pandas package for reading and loading data directly from yahoo etc.
        instruction for the package is below:
        https://pandas-datareader.readthedocs.io/en/latest/remote_data.html#yahoo-finance

        """
        import pandas_datareader.data as web
        import datetime
        start = datetime.datetime(2010, 1, 1)
        end = datetime.datetime(2013, 1, 27)
        f = web.DataReader("F", 'yahoo', start, end)
        print f.ix['2010-01-04']

        # !/usr/bin/env python
        import matplotlib.pyplot as plt
        from matplotlib.dates import DateFormatter, WeekdayLocator, \
            DayLocator, MONDAY
        from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc

        # (Year, month, day) tuples suffice as args for quotes_historical_yahoo
        date1 = (2017, 1, 23)
        date2 = (2017, 1, 27)

        mondays = WeekdayLocator(MONDAY)  # major ticks on the mondays
        alldays = DayLocator()  # minor ticks on the days
        weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
        dayFormatter = DateFormatter('%d')  # e.g., 12

        quotes = quotes_historical_yahoo_ohlc('TSLA', date1, date2)
        for quote in quotes:
            print quote
        print '--'*10
        if len(quotes) == 0:
            raise SystemExit

        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        ax.xaxis.set_major_locator(mondays)
        ax.xaxis.set_minor_locator(alldays)
        ax.xaxis.set_major_formatter(weekFormatter)
        # ax.xaxis.set_minor_formatter(dayFormatter)

        # plot_day_summary(ax, quotes, ticksize=3)
        candlestick_ohlc(ax, quotes, width=0.6)

        ax.xaxis_date()
        ax.autoscale_view()
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

        plt.show()





def showStock2(StockName = 'SPY', DaysBack=20):
    stockname = StockName
    StockInterested = Stock(stockname=stockname)
    StockInterested.loadDataFromPandas()



def showStock(StockName = 'NASDAQ:ADBE', DaysBack=20, ExportFig = True, ShowFig = True):


    # from datetime import datetime
    # import pandas_datareader.data as web
    # import matplotlib.finance
    # data = web.DataReader(StockName, 'yahoo', datetime(2007, 10, 1), datetime(2009, 4, 1))
    # print data
    #candlestick2_ochl(ax, data['Open'], data['High'], data['Low'], data['Close'])

    import datetime
    [year, month, day] = datetime.date.today().isoformat().split('-')
    today = [str(int(month)), str(int(day)), year]
    print today

    stockname = StockName
    StockInterested = Stock(stockname = stockname, eDate=today)
    StockInterested.addMidOpenClose()
    StockInterested.addCloseSubOpen()
    StockInterested.addDailyVariationPercentage()
    StockInterested.addMidHighLow()
    StockInterested.addHighSubLow()
    StockInterested.addMaxDailyVariationPercentage()
    StockInterested.add2ndDayOpenSub1stClose()
    StockInterested.addOpeningRisePercentage()
    #StockInterested.showDataList()
    print StockInterested.StockData.head(10)
    #StockInterested.showGraph()
    StockInterestedAfter1dayAverage, std1 = StockInterested.averageGivenPeriod(1)
    StockInterestedAfter5dayAverage, std5 = StockInterested.averageGivenPeriod(5)
    StockInterestedAfter20Average, std20 = StockInterested.averageGivenPeriod(20)

    fig = plt.figure(figsize=(8,11))
    ax1 = fig.add_subplot(111)
    ax1.plot(StockInterestedAfter1dayAverage,'k-o', markersize=2)
    ax1.plot(StockInterestedAfter5dayAverage, 'b-')
    ax1.plot(StockInterestedAfter20Average, 'r-')
    ax1.plot(np.array(StockInterestedAfter20Average) - 2 * np.array(std20), 'y-')
    ax1.plot(np.array(StockInterestedAfter20Average) + 2 * np.array(std20), 'g-')
    ax1.grid(b=True, which='both', color='0.65',linestyle='-')




    daysback = DaysBack
    x = np.linspace(0, daysback-1, daysback)
    y = StockInterestedAfter1dayAverage[0:daysback]

    A = np.vstack([x, np.ones(len(x))]).T
    print "---"*5
    print np.linalg.lstsq(A, y)
    print "---" * 5
    ans = np.linalg.lstsq(A, y)
    m, c = ans[0]
    residues = ans[1][0]
    print residues
    print '====='
    print(m, c)
    mksize = 2
    rat = 1.5
    if daysback < 50:
        mksize = 10
        rat = 2

    ax1.plot(x, y, 'o-', label='Original data', markersize=mksize)
    ax1.plot(x, m * x + c, 'r', label='Fitted line')
    ax1.set_xlim([0, daysback])
    mid2 = (min(m * x + c) + max(m * x + c))/2
    mid = (min(y) + max(y))/2
    diff = max(y) - min(y)
    diff2 = max(m * x + c) - min(m * x + c)
    percent = diff/mid2

    ax1.set_ylim([mid2-(diff)/2*rat, mid2+(diff)/2*rat])


    #plt.legend()
    ax1.invert_xaxis() #gca().invert_xaxis()
    ax2 = ax1.twinx()
    ax2.set_ylim([-percent/2*rat, +percent/2*rat])
    ax2.yaxis.set_major_locator(MaxNLocator(20))
    ax1.set_title(StockName+': '+ str(datetime.date.today()) +': ' +str(round(residues/(mid2**2), 6)))
    if ExportFig == True:
        filepath = '/Users/Tao/Google Drive/StockPlots/'
        filename = StockName + str(DaysBack) +'.png'
        filename2 = StockName + str(DaysBack) + '.html'
        plt.savefig(filepath+filename)
        #mpld3.save_html(fig, filepath+filename2)
    if ShowFig == True:
        plt.show()
    return plt.figure()





def main():

    """showStock2()"""
    #showStock(StockName='SPY')

    DataBase().UpdateStocks()




    StockList = ['^IXIC', 'NDAQ', 'SPY', 'JNPR', 'WMT', 'MAT', 'ZG', 'AMAT', 'EBAY', 'GOLD', 'VMW', 'NVDA','TSLA']
    GenlStockList = ['NDAQ', 'SPY', 'GOLD']

    SemiStockList = [ 'MU', 'NVDA', 'AMD', 'INTC', 'AMAT','ASML', 'KLAC', 'COHR', 'QCOM' , 'VIAV', 'IIVI']
    EcomStockList = ['BABA', 'AMZN', 'EBAY', 'PYPL']
    CloudStockList = ['GOOG','GOOGL','VMW', 'JNPR','TWTR','MSFT', 'NOW', 'SPLK', 'SAP', 'PYPL', 'IBM', 'NOW', 'BABA']
    CarStockList = ['TSLA', 'F', 'GM', 'FCAU']
    BioMedStockList = ['JNJ','ILMN', 'WBA']
    LaserStockList = ['OCLR', 'IPGP', 'COHR']
    AIStockList = []
    NewStockList = ['CLDR', 'SNAP']


    GoodStockList1 = ['NDAQ', 'SPY', 'GOLD','VMW','NVDA', 'AMD','COHR', 'NOW', 'PYPL', 'BABA']
    GoodStockList2 = [ 'JNPR', 'ASML','LRCX','AMAT', 'TSLA', 'AAPL']
    GoodStockList3 = ['^IXIC', 'FCAU']
    GoodStockList4 =['MU', 'IPGP']

    FinanceStockList = ['BAC','GS']



    #AllStockList = GenlStockList + SemiStockList \
    AllStockList = StockList + GenlStockList + EcomStockList \
                   + CloudStockList + CarStockList + BioMedStockList + LaserStockList + SemiStockList

    AllStockList = list(set(AllStockList))
    print AllStockList
    AllStockList= ['GOOG', 'GOOGL', 'TSLA', 'IIVI', 'AMAT', 'NVDA', 'WBA', 'COHR', 'GM', 'MSFT', 'IBM', 'SPY', 'GOLD', 'TWTR', 'SPLK', 'FCAU', 'AMD', '^IXIC', 'ASML', 'BABA', 'OCLR', 'INTC', 'F', 'PYPL', 'ZG', 'WMT', 'ILMN', 'QCOM', 'JNJ', 'AMZN', 'MAT', 'IPGP', 'EBAY', 'MU', 'VIAV', 'NDAQ', 'JNPR', 'VMW', 'NOW', 'KLAC']


    MostStockList = ['AMZN', 'AMAT'] #['IPGP','NVDA','VMW', 'GOLD', 'COHR']

    import matplotlib.pyplot as plt2, mpld3
    #plt2.plot([3, 1, 4, 1, 5], 'ks-', mec='w', mew=5, ms=20)
    #mpld3.show()


    DaysBackList = [10, 20, 50, 200]
    for Stock in AllStockList: #GoodStockList1+GoodStockList2+GoodStockList3:
        S = DeriveMetrics(StockName_=Stock)
        for DaysBack in DaysBackList:
            print Stock +':' + str(DaysBack)
            #fig = showStock(StockName=Stock, DaysBack=DaysBack, ShowFig=False)
            NewPlot(Stock=S, DaysBack_=DaysBack, ShowFig=False, PrintFlag=False)
        #htmlfig = mpld3.fig_to_html(fig)
        #print htmlfig

    StockGrowthCompareList = AllStockList#['^IXIC', 'COHR', 'PYPL', 'IPGP', 'NVDA', 'AMD', 'AMAT', 'KLAC', 'MU']
    DaysBackList = [5, 15, 20, 50, 200]
    SList = [DeriveMetrics(StockName_=SName) for SName in StockGrowthCompareList]
    for DaysBack in DaysBackList:
        fig = plt.figure(figsize=(8, 14))
        ax1 = fig.add_subplot(211)
        MultiStockGrowthComparison(StockInstanceList_=SList, DaysBack_=DaysBack, ExportFig=True, ShowFig=False, figsettings=(fig,ax1))
        ax2 = fig.add_subplot(212)
        MultiStockBollingerPhaseComparison(StockInstanceList_=SList, DaysBack_=DaysBack, ExportFig=True, ShowFig=False, figsettings=(fig,ax2))


def UsingStockStats(StockName_):
    from stockstats import *
    import datetime
    [year, month, day] = datetime.date.today().isoformat().split('-')
    today = [str(int(month)), str(int(day)), year]
    print today
    StockName = StockName_

    stockname = StockName
    S = Stock(stockname=stockname, eDate=today)

    stock = StockDataFrame.retype(S.StockData)
    stock.get('boll')
    print '==========This is from stock data frame: \n', stock



def DeriveMetrics(StockName_):

    [year, month, day] = datetime.date.today().isoformat().split('-')
    today = [str(int(month)), str(int(day)), year]
    print today
    StockName = StockName_

    stockname = StockName
    S = Stock(stockname=stockname, eDate=today, DataSource='DataBase')

    S.addDaysBack(name='DaysBack')
    S.addGainLossLabel()
    S.addRollingLstSq(name='k', days=5)
    S.addRollingLstSq(name='bias', days=5)
    S.addRollingLstSq(name='k', days=10)
    S.addRollingLstSq(name='bias', days=10)
    S.addRollingLstSq(name='k', days=20)
    S.addRollingLstSq(name='bias', days=20)
    S.addRollingLstSq(name='k', days=50)
    S.addRollingLstSq(name='bias', days=50)
    S.addRollingLstSq(name='k', days=200)
    S.addRollingLstSq(name='bias', days=200)
    S.addRollingAverage(name='5DMovAve', days=5)
    S.addRollingAverage(name='20DMovAve', days=20)
    S.addRollingAverage(name='200DMovAve', days=200)
    S.addRollingStd(name='UpBlg', days=20, sign=2)
    S.addRollingStd(name='LwBlg', days=20, sign=-2)
    S.addRollingNormalizedValue(name='NormalizedBollinger', days=20, sign=2)
    S.addRollingRSI()
    S.addRollingStcOsc(name='Stochastic Osc', days=15, D=3)
    return S


def NewPlot(Stock, DaysBack_=50, ExportFig = True, ShowFig = True, PrintFlag = True):

    S = Stock

    # x = np.linspace(-S.StockData.shape[0] + 1, 0, S.StockData.shape[0])
    # print 'last 20 x is'
    # print x[-20:-1]
    plotoffset = 1
    xAxis = S.StockData['DaysBack']
    fig = plt.figure(figsize=(8, 14))
    ax1 = fig.add_subplot(111)

    candlestick_ohlc(ax1, S.StockData[['DaysBack', 'Open','High', 'Low', 'Close']].values, width=.75, colorup='g', colordown='r',alpha=0.75)
    #ax1.plot(xAxis, S.StockData['Close'],'k-o', markersize=2)
    ax1.plot(xAxis, S.StockData['5DMovAve'],'b-')
    ax1.plot(xAxis, S.StockData['20DMovAve'],'r-')
    ax1.plot(xAxis, S.StockData['200DMovAve'],'k-')
    ax1.plot(xAxis, S.StockData['20DMovAve'] + S.StockData['UpBlg'], 'y-')
    ax1.plot(xAxis, S.StockData['20DMovAve'] + S.StockData['LwBlg'], 'g-')
    ax1.grid(b=True, which='both', color='0.65', linestyle='-')

    sDate = 0
    eDate = DaysBack_
    ax1.set_xlim([xAxis[eDate-1], xAxis[sDate]+plotoffset])

    daysback = eDate - sDate
    mksize = 2
    rat = 1.5
    if daysback < 50:
        mksize = 10
        rat = 2

    x =  S.StockData['DaysBack'].ix[sDate:eDate].values[::-1]
    y = S.StockData['LstSqk_'+str(eDate-sDate)+'Days'][eDate-1] * x + S.StockData['LstSqbias_'+str(eDate-sDate)+'Days'][eDate-1]
    #print y
    #print S.StockData['Date'][0:20][::-1]
    #ax1.plot(xAxis, S.StockData['Close'], 'ko', label='Original data', markersize=mksize)
    ax1.plot(xAxis[sDate:eDate][::-1], y, 'r-', label='Fitted line', markersize=1)
    ax1.set_title(S.stockName + ': ' + str(datetime.date.today()) + ': ' + str('round(residues / (mid2 ** 2), 6)'))



    maxStock = max(S.StockData['Close'][sDate:eDate])
    minStock = min(S.StockData['Close'][sDate:eDate])
    mid = (minStock + maxStock) / 2
    mid2 = (min(y) + max(y)) / 2
    diff = maxStock - minStock
    diff2 = max(y) - min(y)
    percent = diff / mid2
    offset = (S.StockData['Close'][sDate] - (mid2 - (diff) / 2 * rat)) / mid2

    ax1.set_ylim([mid2 - (diff) / 2 * rat, mid2 + (diff) / 2 * rat])

    ax2 = ax1.twinx()
    ax2.set_ylim([-percent / 2 * rat , percent / 2 * rat] )
    ax2.yaxis.set_major_locator(MaxNLocator(20))

    ax3 = fig.add_subplot(414)
    ax3.set_xlim([xAxis[eDate - 1], xAxis[sDate]+plotoffset])

    #ax1.plot(xAxis, S.StockData['LstSqk_10Days']*100)
    ax3.plot(xAxis, S.StockData['14-Day RSI'], 'o-')
    ax3.plot(xAxis, np.ones(xAxis.shape) * 100, '-')
    ax3.plot(xAxis, np.ones(xAxis.shape) * 70, 'r-')
    ax3.plot(xAxis, np.ones(xAxis.shape) * 50, 'g-')
    ax3.plot(xAxis, np.ones(xAxis.shape) * 30, 'k-')
    ax3.plot(xAxis, np.ones(xAxis.shape) * 0, '-.')
    #ax1.plot(xAxis, S.StockData['14-Day RSI'])

    ax3.plot(xAxis, S.StockData['StcOsc K%']*100, 'k-.')
    ax3.plot(xAxis, S.StockData['StcOsc D%']*100, 'k-')
    ax3.plot(xAxis, S.StockData['StcOsc D%D%'] * 100, 'r-')

    #plt.figure()

    #ax3.plot(xAxis, (S.StockData['NormalizedBollinger'] + np.ones(xAxis.shape))*50, 'g-o')
    ##ax1.plot(xAxis, S.StockData['StcOsc K%']*20 - S.StockData['StcOsc D%'] * 20+np.ones(xAxis.shape) * 175, 'y-o')
    ##ax1.plot(xAxis, np.ones(xAxis.shape) * 175, 'y-')
    #ax1.grid()
    #ax1.ylim(120,190)
    #plt.show()

    """

    plt.figure()
    plt.plot(xAxis, (S.StockData['StcOsc K%'] - S.StockData['StcOsc D%'])*20 , 'y-o')
    #plt.plot(xAxis, np.ones(xAxis.shape) * 175, 'y-')
    plt.plot(xAxis, S.StockData['NextDayGain|Loss%']*5, 'r-o')
    plt.xlim(xAxis[eDate - 1], xAxis[sDate])
    plt.ylim(-20, 20)
    plt.grid()

    # plt.plot(xAxis, S.StockData['LstSqk_20Days'])
    plt.show()
    #print (S.StockData['Date'].diff())#.dt.days())#[0] - S.StockData['Date'][1]).days()#total_seconds()
    """
    #plt.figure()
    #plt.plot(S.StockData['NextDayGain|Loss%'].apply(lambda x: int(x>0)), S.StockData['StcOsc K%-D%'].apply(lambda x: int(x>0)),'o')
    # print sum((S.StockData['NextDayGain|Loss%'].apply(lambda x: int(x > 0)) + S.StockData['StcOsc K%-D%'].apply(lambda x: int(x > 0)))==0)
    # print sum((S.StockData['NextDayGain|Loss%'].apply(lambda x: int(x > 0)) + S.StockData['StcOsc K%-D%'].apply(lambda x: int(x > 0)))==2)
    # print sum((S.StockData['NextDayGain|Loss%'].apply(lambda x: int(x > 0)) + S.StockData['StcOsc K%-D%'].apply(lambda x: int(x > 0))) >-10)


    #plt.grid()
    #plt.show()
    if PrintFlag == True:
        print S.StockData#['LstSqk_20days']
    #S.StockData.to_csv(S.stockName+'.csv')
    #print sum(S.StockData['NextDayGain|Loss%'][0:100] * S.StockData['StcOsc K%-D%'][0:100] > 0)
    #print sum(S.StockData['NextDayGain|Loss%'][0:100] * S.StockData['StcOsc K%-D%'][0:100] < 0)

    ##########################################


    if ExportFig == True:
        filepath = '/Users/Tao/Google Drive/StockPlots/'
        filename = S.stockName + str(DaysBack_) + '_.png'
        filename2 = S.stockName + str(DaysBack_) + '_.html'
        plt.savefig(filepath + filename)
        # mpld3.save_html(fig, filepath+filename2)
    if ShowFig == True:
       plt.show()
    plt.close('all')
    #return plt.figure()

def test():
    S = DeriveMetrics('COHR')
    ReturnVsVolumn(Stock=S)
    S = DeriveMetrics('AAPL')
    ReturnVsVolumn(Stock=S)

def ReturnVsVolumn(Stock, DaysBack_=50, ExportFig = True, ShowFig = True, PrintFlag = True):

    S = Stock


    plotoffset = 1
    xAxis = S.StockData['DaysBack']
    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)
    print S.StockData
    ax1.plot(xAxis, np.log(np.abs(S.StockData['NextDayGain|Loss%'])), label='return%')
    ax1.plot(xAxis[1::], -np.diff(np.log(S.StockData['Volume'])),label='Volumn')

    #ax1.plot((S.StockData['Volume']), np.abs(S.StockData['NextDayGain|Loss%']).shift(1), '.')
    #ax1.plot((S.StockData['Volume']), (S.StockData['NormalizedBollinger']), '.')
    #ax1.plot(np.diff(np.log(S.StockData['Volume'])),np.log(np.abs(S.StockData['NextDayGain|Loss%']))[0:-1],  '.')

    plt.show()

    # candlestick_ohlc(ax1, S.StockData[['DaysBack', 'Open','High', 'Low', 'Close']].values, width=.75, colorup='g', colordown='r',alpha=0.75)
    # #ax1.plot(xAxis, S.StockData['Close'],'k-o', markersize=2)
    # ax1.plot(xAxis, S.StockData['5DMovAve'],'b-')
    # ax1.plot(xAxis, S.StockData['20DMovAve'],'r-')
    # ax1.plot(xAxis, S.StockData['200DMovAve'],'k-')
    # ax1.plot(xAxis, S.StockData['20DMovAve'] + S.StockData['UpBlg'], 'y-')
    # ax1.plot(xAxis, S.StockData['20DMovAve'] + S.StockData['LwBlg'], 'g-')
    # ax1.grid(b=True, which='both', color='0.65', linestyle='-')
    #
    # sDate = 0
    # eDate = DaysBack_
    # ax1.set_xlim([xAxis[eDate-1], xAxis[sDate]+plotoffset])
    #
    # daysback = eDate - sDate
    # mksize = 2
    # rat = 1.5
    # if daysback < 50:
    #     mksize = 10
    #     rat = 2
    #
    # x =  S.StockData['DaysBack'].ix[sDate:eDate].values[::-1]
    # y = S.StockData['LstSqk_'+str(eDate-sDate)+'Days'][eDate-1] * x + S.StockData['LstSqbias_'+str(eDate-sDate)+'Days'][eDate-1]
    # #print y
    # #print S.StockData['Date'][0:20][::-1]
    # #ax1.plot(xAxis, S.StockData['Close'], 'ko', label='Original data', markersize=mksize)
    # ax1.plot(xAxis[sDate:eDate][::-1], y, 'r-', label='Fitted line', markersize=1)
    # ax1.set_title(S.stockName + ': ' + str(datetime.date.today()) + ': ' + str('round(residues / (mid2 ** 2), 6)'))
    #
    #
    #
    # maxStock = max(S.StockData['Close'][sDate:eDate])
    # minStock = min(S.StockData['Close'][sDate:eDate])
    # mid = (minStock + maxStock) / 2
    # mid2 = (min(y) + max(y)) / 2
    # diff = maxStock - minStock
    # diff2 = max(y) - min(y)
    # percent = diff / mid2
    # offset = (S.StockData['Close'][sDate] - (mid2 - (diff) / 2 * rat)) / mid2
    #
    # ax1.set_ylim([mid2 - (diff) / 2 * rat, mid2 + (diff) / 2 * rat])
    #
    # ax2 = ax1.twinx()
    # ax2.set_ylim([-percent / 2 * rat , percent / 2 * rat] )
    # ax2.yaxis.set_major_locator(MaxNLocator(20))
    #
    # ax3 = fig.add_subplot(414)
    # ax3.set_xlim([xAxis[eDate - 1], xAxis[sDate]+plotoffset])
    #
    # #ax1.plot(xAxis, S.StockData['LstSqk_10Days']*100)
    # ax3.plot(xAxis, S.StockData['14-Day RSI'], 'o-')
    # ax3.plot(xAxis, np.ones(xAxis.shape) * 100, '-')
    # ax3.plot(xAxis, np.ones(xAxis.shape) * 70, 'r-')
    # ax3.plot(xAxis, np.ones(xAxis.shape) * 50, 'g-')
    # ax3.plot(xAxis, np.ones(xAxis.shape) * 30, 'k-')
    # ax3.plot(xAxis, np.ones(xAxis.shape) * 0, '-.')
    # #ax1.plot(xAxis, S.StockData['14-Day RSI'])
    #
    # ax3.plot(xAxis, S.StockData['StcOsc K%']*100, 'k-.')
    # ax3.plot(xAxis, S.StockData['StcOsc D%']*100, 'k-')
    # ax3.plot(xAxis, S.StockData['StcOsc D%D%'] * 100, 'r-')
    #
    # #plt.figure()
    #
    # #ax3.plot(xAxis, (S.StockData['NormalizedBollinger'] + np.ones(xAxis.shape))*50, 'g-o')
    # ##ax1.plot(xAxis, S.StockData['StcOsc K%']*20 - S.StockData['StcOsc D%'] * 20+np.ones(xAxis.shape) * 175, 'y-o')
    # ##ax1.plot(xAxis, np.ones(xAxis.shape) * 175, 'y-')
    # #ax1.grid()
    # #ax1.ylim(120,190)
    # #plt.show()
    #
    # """
    #
    # plt.figure()
    # plt.plot(xAxis, (S.StockData['StcOsc K%'] - S.StockData['StcOsc D%'])*20 , 'y-o')
    # #plt.plot(xAxis, np.ones(xAxis.shape) * 175, 'y-')
    # plt.plot(xAxis, S.StockData['NextDayGain|Loss%']*5, 'r-o')
    # plt.xlim(xAxis[eDate - 1], xAxis[sDate])
    # plt.ylim(-20, 20)
    # plt.grid()
    #
    # # plt.plot(xAxis, S.StockData['LstSqk_20Days'])
    # plt.show()
    # #print (S.StockData['Date'].diff())#.dt.days())#[0] - S.StockData['Date'][1]).days()#total_seconds()
    # """
    # #plt.figure()
    # #plt.plot(S.StockData['NextDayGain|Loss%'].apply(lambda x: int(x>0)), S.StockData['StcOsc K%-D%'].apply(lambda x: int(x>0)),'o')
    # # print sum((S.StockData['NextDayGain|Loss%'].apply(lambda x: int(x > 0)) + S.StockData['StcOsc K%-D%'].apply(lambda x: int(x > 0)))==0)
    # # print sum((S.StockData['NextDayGain|Loss%'].apply(lambda x: int(x > 0)) + S.StockData['StcOsc K%-D%'].apply(lambda x: int(x > 0)))==2)
    # # print sum((S.StockData['NextDayGain|Loss%'].apply(lambda x: int(x > 0)) + S.StockData['StcOsc K%-D%'].apply(lambda x: int(x > 0))) >-10)
    #
    #
    # #plt.grid()
    # #plt.show()
    # if PrintFlag == True:
    #     print S.StockData#['LstSqk_20days']
    # S.StockData.to_csv(S.stockName+'.csv')
    # #print sum(S.StockData['NextDayGain|Loss%'][0:100] * S.StockData['StcOsc K%-D%'][0:100] > 0)
    # #print sum(S.StockData['NextDayGain|Loss%'][0:100] * S.StockData['StcOsc K%-D%'][0:100] < 0)
    #
    # ##########################################
    #
    #
    # if ExportFig == True:
    #     filepath = '/Users/Tao/Google Drive/StockPlots/'
    #     filename = S.stockName + str(DaysBack_) + '_.png'
    #     filename2 = S.stockName + str(DaysBack_) + '_.html'
    #     plt.savefig(filepath + filename)
    #     # mpld3.save_html(fig, filepath+filename2)
    # if ShowFig == True:
    #    plt.show()
    # return plt.figure()





def MultiStockPlot(StockNameList_, DaysBack_=50, ExportFig = True, ShowFig = True):

    SList = [DeriveMetrics(StockName_=SName) for SName in StockNameList_]

    sDate = 0
    eDate = DaysBack_

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    # for S in SList:
    #     xAxis = S.StockData['DaysBack']
    #     ax1.plot(xAxis, S.StockData['NextDayGain|Loss%'],'-o', markersize=2)
    #     ax1.set_xlim([xAxis[eDate - 1], xAxis[sDate]])


    x0 = 5
    x = SList[0].StockData['NextDayGain|Loss%'][0:DaysBack_]
    for S in SList[1::]:
        y = S.StockData['NextDayGain|Loss%'][0:DaysBack_]
        ax1.plot(x, y, '.', label=S.stockName)
        ax1.set_xlim([-x0, x0])
        ax1.set_ylim([-x0, x0])

    ax1.grid(b=True, which='both', color='0.65', linestyle='-')
    ax1.set_title('+'.join(StockNameList_) + ': ' + str(datetime.date.today()) + ': ' + str('round(residues / (mid2 ** 2), 6)'))
    plt.legend()
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    ax1 = fig.add_subplot(111)

    #x0 = 5
    x = SList[0].StockData['Date'][0:DaysBack_]
    for S in SList:
        y = S.StockData['Close'][0:DaysBack_]/S.StockData['Close'][DaysBack_]
        ax1.plot(x, y, 'o-', label=S.stockName)
        #ax1.set_xlim([-x0, x0])
        #ax1.set_ylim([-x0, x0])

    ax1.grid(b=True, which='both', color='0.65', linestyle='-')
    ax1.set_title(
        '+'.join(StockNameList_) + ': ' + str(datetime.date.today()) + ': ' + str('round(residues / (mid2 ** 2), 6)'))
    plt.legend()

    plt.show()

    # if ExportFig == True:
    #     filepath = '/Users/Tao/Google Drive/StockPlots/'
    #     filename = StockName + str(DaysBack_) + '_.png'
    #     filename2 = StockName + str(DaysBack_) + '_.html'
    #     plt.savefig(filepath + filename)
    #     # mpld3.save_html(fig, filepath+filename2)
    # if ShowFig == True:
    #    plt.show()
    # return plt.figure()


def MultiStockGrowthComparison(StockInstanceList_, DaysBack_=50, ExportFig=True, ShowFig=True, figsettings=None):


    SList = StockInstanceList_

    sDate = 0
    eDate = DaysBack_
    if figsettings == None:
        fig = plt.figure(figsize=(8, 11))
        ax1 = fig.add_subplot(111)
    fig, ax1 = figsettings

    import operator
    StockDict = {}
    for S in SList:
        StockDict[S] = S.StockData['Close'][0] / S.StockData['Close'][DaysBack_ - 1]
    Top8SList = sorted(StockDict.items(), key=operator.itemgetter(1))[-8:-1]

    # x0 = 5
    #x = SList[0].StockData.index[0:DaysBack_]
    x = range(0, -DaysBack_, -1)
    for S, lasty in Top8SList:
        y = S.StockData['Close'][0:DaysBack_] / S.StockData['Close'][DaysBack_-1]
        ax1.plot(x, y, 'o-', label=S.stockName)
        # ax1.set_xlim([-x0, x0])
        # ax1.set_ylim([-x0, x0])
    StockNameList_ = [S.stockName for S, lasty in Top8SList]
    ax1.grid(b=True, which='both', color='0.65', linestyle ='-')
    ax1.set_title(
        '+'.join(StockNameList_) + ': ' + str(datetime.date.today()) + ': ' + str(
            'round(residues / (mid2 ** 2), 6)'))
    plt.legend(loc=2)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    #plt.show()

    if ExportFig == True:
        filepath = '/Users/Tao/Google Drive/StockPlots/'
        filename = 'MultiStockGrowthComparison' + str(DaysBack_) + '_.png'
        #filename2 = StockName + str(DaysBack_) + '_.html'
        plt.savefig(filepath + filename)
        # mpld3.save_html(fig, filepath+filename2)
    if ShowFig == True:
       plt.show()
    return plt.figure()

def MultiStockBollingerPhaseComparison(StockInstanceList_, DaysBack_=50, ExportFig=True, ShowFig=True, figsettings=None):

    SList = StockInstanceList_ #[DeriveMetrics(StockName_=SName) for SName in StockNameList_]
    sDate = 0
    eDate = DaysBack_


    if figsettings == None:
        fig = plt.figure(figsize=(28, 11))
        ax1 = fig.add_subplot(111)
    fig, ax1 = figsettings


    # x0 = 5
    x = SList[0].StockData['DaysBack'][0:DaysBack_]
    yaverage = np.zeros(x.shape)
    for S in SList:
        y = S.StockData['NormalizedBollinger'][0:DaysBack_]
        y = np.nan_to_num(y)
        yaverage = y + yaverage

        ax1.plot(x, y, 'o-', label=S.stockName)
        # ax1.set_xlim([-x0, x0])
        # ax1.set_ylim([-x0, x0])

    yaverage = yaverage / len(SList)
    ax1.plot(x, yaverage, 'r-', label='Average', linewidth=5)


    # x = SList[0].StockData['DaysBack'][0:DaysBack_]
    # for S in SList:
    #     y = S.StockData['NormalizedBollinger'][0:DaysBack_]
    #     ax1.plot(x, y, 'o-', label=S.stockName)
    #     # ax1.set_xlim([-x0, x0])
    #     # ax1.set_ylim([-x0, x0])


    ax1.plot(x, np.ones(x.shape), 'y-')
    ax1.plot(x, -np.ones(x.shape), 'r-')


    StockNameList_ = [S.stockName for S in SList]
    ax1.grid(b=True, which='both', color='0.65', linestyle ='-')
    ax1.set_title(
        '+'.join(StockNameList_) + ': ' + str(datetime.date.today()) + ': ' + str(
            'round(residues / (mid2 ** 2), 6)'))
    plt.legend(loc=2)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    #plt.show()

    if ExportFig == True:
        filepath = '/Users/Tao/Google Drive/StockPlots/'
        filename = 'MultiStockBollingerComparison' + str(DaysBack_) + '_.png'
        #filename2 = StockName + str(DaysBack_) + '_.html'
        fig.savefig(filepath + filename)
        # mpld3.save_html(fig, filepath+filename2)
    if ShowFig == True:
       plt.show()
    return plt.figure()

def newtest():
    import pandas as pd
    import numpy as np
    txc = pd.DataFrame(np.random.rand(10, 2), columns=['sql_resp', 'tran_count'])

    print txc

    def roll(df, w, **kwargs):
        roll_array = np.dstack([df.values[i:i + w, :] for i in range(len(df.index) - w + 1)]).T
        panel = pd.Panel(roll_array,
                         items=df.index[w - 1:],
                         major_axis=df.columns,
                         minor_axis=pd.Index(range(w), name='roll'))
        return panel.to_frame().unstack().T.groupby(level=0, **kwargs)

    def myfunction(x, y):
        print x
        x0 = x[y[0]]
        x1 = x[y[1]]
        a = np.ma.corrcoef(x0, x1)
        print a[0, 1]
        return a[0, 1]

    df = txc
    print df
    df['newc'] = roll(df, 3).apply(myfunction, ('sql_resp', 'tran_count'))
    print '--' * 10

def twoStockCorrelation(StockNameList_, nDays=5, kShift=0):
    import pandas as pd
    import numpy as np
    # txc = pd.DataFrame(np.random.rand(10, 2), columns=['sql_resp', 'tran_count'])
    #
    # print txc

    def roll(df, w, **kwargs):
        roll_array = np.dstack([df.values[i:i + w, :] for i in range(len(df.index) - w + 1)]).T
        panel = pd.Panel(roll_array,
                         items=df.index[w - 1:],
                         major_axis=df.columns,
                         minor_axis=pd.Index(range(w), name='roll'))
        return panel.to_frame().unstack().T.groupby(level=0, **kwargs)

    def myfunction(x, y):
        x0 = x[y[0]]
        x1 = x[y[1]]
        a = np.ma.corrcoef(x0, x1)*200
        #print a[0, 1]
        return a[0, 1]


    SList = [DeriveMetrics(StockName_=SName) for SName in StockNameList_]
    df = pd.DataFrame()
    df[SList[0].stockName+'0_close'] = SList[0].StockData['Close']
    df[SList[1].stockName+'1_close'] = SList[1].StockData['Close'].shift(kShift)

    DF = pd.DataFrame(df.fillna(value=0.001), columns=df.columns)
    df[SList[0].stockName+'-'+SList[1].stockName+'_'+'corrcoef'] = pd.DataFrame(roll(DF, w=nDays).apply(myfunction, (df.columns[0], df.columns[1])),dtype=np.float)
    print '--' * 10
    print df
    df.plot()
    plt.show()
    return df

def MultiStockCorrelation(StockNameList_, nDays_=50, kShift=2, ExportFig=True, ShowFig=True):

    SList = [DeriveMetrics(StockName_=SName) for SName in StockNameList_]

    sDate = 0
    eDate = DaysBack_
    fig = plt.figure(figsize=(28, 11))
    ax1 = fig.add_subplot(111)

    # x0 = 5

    x = SList[0].StockData['DaysBack'][0:DaysBack_]
    for S in SList:
        y = S.StockData['NormalizedBollinger'][0:DaysBack_]
        ax1.plot(x, y, 'o-', label=S.stockName)
        # ax1.set_xlim([-x0, x0])
        # ax1.set_ylim([-x0, x0])



    ax1.plot(x, np.ones(x.shape), 'y-')
    ax1.plot(x, -np.ones(x.shape), 'r-')


    ax1.grid(b=True, which='both', color='0.65', linestyle ='-')
    ax1.set_title(
        '+'.join(StockNameList_) + ': ' + str(datetime.date.today()) + ': ' + str(
            'round(residues / (mid2 ** 2), 6)'))
    plt.legend(loc=2)
    plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')

    #plt.show()

    if ExportFig == True:
        filepath = '/Users/Tao/Google Drive/StockPlots/'
        filename = 'MultiStockBollingerComparison' + str(DaysBack_) + '_.png'
        #filename2 = StockName + str(DaysBack_) + '_.html'
        plt.savefig(filepath + filename)
        # mpld3.save_html(fig, filepath+filename2)
    if ShowFig == True:
       plt.show()
    return plt.figure()




if __name__=='__main__':

    # x = datetime.today()
    # y = x.replace(day=x.day + 0, hour=8, minute=44, second=10, microsecond=0)
    # delta_t = y - x
    #
    # secs = delta_t.seconds + 1
    #
    #
    # def hello_world():
    #     print "hello world"
    #
    # print "start"
    # minutes = 50
    # hours = 10
    # while True:
    #     minutes += 1
    #     minutes %= 59
    #     hours = hours + 1 if (minutes == 0) else hours
    #     y = x.replace(day=x.day + 0, hour=hours, minute=minutes, second=0, microsecond=0)
    #     delta_t = y - x
    #
    #     secs = delta_t.seconds + 1
    #     t = Timer(secs, hello_world)
    #     t.start()

    main()
    #NewPlot(StockName_='COHR', DaysBack_=200)
    #MultiStockPlot(StockNameList_=['^IXIC', 'COHR', 'PYPL', 'NVDA', 'AMD', 'INTC', 'AMAT','ASML', 'KLAC', 'COHR', 'QCOM' , 'VIAV', 'IIVI'], DaysBack_=450, ExportFig=True, ShowFig=True)
    #MultiStockPlot(StockNameList_=['AMD', 'NVDA'], DaysBack_=200, ExportFig=True, ShowFig=True)
    #list1 = ['^IXIC', 'COHR', 'PYPL', 'IPGP','NVDA', 'AMD', 'AMAT', 'GOLD']
    #list2 = ['^IXIC', 'COHR', 'PYPL', 'IPGP','NVDA', 'AMD', 'INTC', 'AMAT','ASML', 'KLAC', 'COHR', 'QCOM' , 'VIAV', 'IIVI']
    #list3 = ['ASML', 'KLAC', 'COHR', 'QCOM' , 'VIAV', 'IIVI']

    #MultiStockGrowthComparison(StockNameList_=list2, DaysBack_=5, ExportFig=True, ShowFig=True)
    #MultiStockBollingerPhaseComparison(StockNameList_=list2, DaysBack_=450, ExportFig=True, ShowFig=True)

    #UsingStockStats(StockName_='COHR')
    #NewPlot(StockName_='COHR', DaysBack_=50, ExportFig=True, ShowFig=True)
    #df = twoStockCorrelation(StockNameList_=['GOLD', 'GOLD'], nDays=100, kShift=1)
    #newtest()
    #test()
    #DataBase().DownloadStocks()
    #DataBase().UpdateStocks()

    