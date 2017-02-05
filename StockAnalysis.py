import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Stock(object):

    def __init__(self, stockname = 'NVDA', sDate=['3', '12', '2000'], eDate=['1', '27', '2017']):
        self.YahooFinanceURL = 'http://real-chart.finance.yahoo.com/table.csv?s='
        self.stockName = stockname
        self.startDate = sDate
        self.endDate = eDate
        self.StockData = self.downloadHistoryCSVviaURL(self.composeYahooFinanceQueryURL())


    
    
    def downloadHistoryCSVviaURL(self, url):
        StockDataFrame = pd.read_csv(url)
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

        
    def UnitTest01(self):
        #url = 'http://real-chart.finance.yahoo.com/table.csv?s=YHOO&d=4&e=13&f=2016&g=d&a=3&b=12&c=1996&ignore=.csv'
        url = self.composeYahooFinanceQueryURL()
        
        print self.downloadHistoryCSVviaURL(url)

    def addDerivatedData(self):
        pass

    def addMidOpenClose(self):
        self.StockData['MidOpenClose'] = \
            (self.StockData['Open']+self.StockData['Close']) / 2.0

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
        TodayOpen, TodayClose, TodayHigh, TodayLow, MA20, StdMA20, MA5, NASQ, DJI,

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



def showStock(StockName = 'SPY', DaysBack=20):


    # from datetime import datetime
    # import pandas_datareader.data as web
    # import matplotlib.finance
    # data = web.DataReader(StockName, 'yahoo', datetime(2007, 10, 1), datetime(2009, 4, 1))
    # print data
    #candlestick2_ochl(ax, data['Open'], data['High'], data['Low'], data['Close'])

    #Stock(stockname = 'JNPR').showGraph()
    stockname = StockName
    StockInterested = Stock(stockname = stockname)
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

    plt.plot(StockInterestedAfter1dayAverage,'k-o')
    plt.plot(StockInterestedAfter5dayAverage, 'b-')
    plt.plot(StockInterestedAfter20Average, 'r-')
    plt.plot(np.array(StockInterestedAfter20Average) - 2 * np.array(std20), 'y-')
    plt.plot(np.array(StockInterestedAfter20Average) + 2 * np.array(std20), 'g-')
    #plt.plot(jump *20,'r-')
    #plt.plot(jv * 30, 'b-')
    plt.grid(b=True, which='both', color='0.65',linestyle='-')


    #plt.ylim([-10, 35])
    #plt.gca().invert_xaxis()
    #plt.show()

    daysback = DaysBack
    x = np.linspace(0, daysback-1, daysback)
    y = StockInterestedAfter1dayAverage[0:daysback]

    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    print(m, c)

    plt.plot(x, y, 'o-', label='Original data', markersize=10)
    plt.plot(x, m * x + c, 'r', label='Fitted line')
    plt.xlim([0, daysback])
    plt.ylim([min(y), max(y)])
    #plt.legend()
    plt.gca().invert_xaxis()
    plt.title(StockName)
    plt.show()


    #a200 = Stock(stockname = stockname).averageGivenPeriod(200)
    """"""
    #a50 = Stock(stockname = stockname).averageGivenPeriod(50)
    #a5 = Stock(stockname = stockname).averageGivenPeriod(10)
    #a1 = Stock(stockname = stockname).averageGivenPeriod(1)
    #print a5
    #arise200 = np.array(a5[0:2500]) - np.array(a200[0:2500])
    
    #jump = - np.diff(((np.array(a5[0:2500]) - np.array(a200[0:2500])* 1.02)>=0).astype(np.float))>0
    
    #a_c = np.array(a5[0:(2500-1)]) * jump
    #a_b = np.array(a5[200:(2500+200-1)]) * jump
    #jv = ((a_b - a_c)/a_b > 0.10).astype(np.float)


    """
    #plt.figure()
    #plt.plot(a1)
    plt.plot(arise200,'k-')
    plt.plot(np.array(a1[0:250])*1.02, 'b-')
    plt.plot(jump *20,'r-')
    plt.plot(jv * 30, 'b-')
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
    plt.xlim([0,252*1])
    #plt.ylim([-10, 35])
    plt.gca().invert_xaxis()
    plt.show()
    """

def main():

    """showStock2()"""
    #showStock(StockName='SPY')
    #showStock(StockName='JNPR')
    ###showStock(StockName='NDAQ')
    #showStock(StockName='BIDU')

    StockList = ['NDAQ', 'SPY', 'JNPR', 'WMT', 'MAT', 'ZG', 'AMAT', 'EBAY', 'GOLD', 'VMW', 'NVDA','TSLA']
    GenlStockList = ['NDAQ', 'SPY', 'GOLD', 'AAPL', 'GOOG','VMW', 'NVDA','TSLA']
    SemiStockList = [ 'NVDA', 'AMD', 'INTC', 'AMAT','ASML', 'KLAC', 'COHR', 'QCOM' ]
    EcomStockList = ['BABA', 'AMZN', 'EBAY', 'PYPL']
    CloudStockList = ['GOOG','GOOGL','YHOO', 'VMW', 'JNPR','TWTR','MSFT']
    CarStockList = ['TSLA', 'F', 'GM', 'FCAU']
    AIStockList = []
    Ab5dayMAStockList = []
    GoodStockList = ['COHR','GOLD','VMW','AMAT','NVDA', 'ASML','LRCX']
    # Current good ones are gold, cohr, vmw, nvda, amat, lrcx

    DownTurnStockList = ['NVDA', 'AMD', 'QCOM', 'BABA', 'AAPL']






    MostStockList = ['TSLA','VMW', 'GM']
    DaysBack = 20
    for Stock in MostStockList:
        showStock(StockName=Stock, DaysBack=DaysBack)

    # showStock(StockName='NDAQ', DaysBack=80)  # likely to go up,
    # showStock(StockName='SPY', DaysBack=40)  # likely to go up,
    # #showStock(StockName='DJI', DaysBack=40)  # likely to go up,
    # showStock(StockName='JNPR', DaysBack=30)  # likely to go up,
    #
    # showStock(StockName='WMT', DaysBack=200)  # likely to go up,
    # showStock(StockName='MAT', DaysBack=200)  # likely to go up,
    # showStock(StockName='ZG', DaysBack=200)  # likely to go up,
    # showStock(StockName='AMAT', DaysBack=200)  # likely to go up,
    # showStock(StockName='EBAY', DaysBack=200)  # likely to go up,
    # showStock(StockName='GOLD', DaysBack=400)  # likely to go up,


    # if gold hasn't rise too much tomorrow morning,
    # and I have some stock to sell then I should buy gold.
    # if gold goes up too much then I shouldn't buy it.,
    # estimate price is about $83
    # it seems gold price grow speed is slowing down a little bit.


    # showStock(StockName='VMW', DaysBack=200)  # next 26-27 will drop
    # showStock(StockName='NVDA', DaysBack=600) #now in grow 26-30 might big
    # showStock(StockName='NVDA', DaysBack=6)  # now in grow 26-30 might big
    # showStock(StockName='TSLA', DaysBack=200) #drop in 26-27, then grow.
    # As show on 27, it starts growing. It might go up to $125
    # or even close to $130 in 1 or 2 days.
    # watch closely and sell it once it reach the top.
    #  showStock(StockName='COHR', DaysBack=60) # next 26-27 will big drop


    ###showStock(StockName='KLAC')
    #showStock(StockName='ASML')
    #showStock(StockName='USDCNY')
    
    
if __name__=='__main__':
    main()
    
    
    
    