# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 21:30:14 2016

@author: Tao
"""

    del buildKBArticleDiction(self, TADFThreshold = 10, TFIDFTTopping=30, TFRankCutoff = 10000):        
        load dataFrame files and then proceed cut off and gather all the left words to build 
        a set of words and save it as a file for future usage. 
        Diet = set() 
        PathName = self.KBArticleTFIDFDataPath 
        #FileNamelndices = range(0, 5, 1) 
        #FileNameList = [PathName + 'ttestData' + str(Filelndex).zfill(5) +'.csv' 
        for FileIndex in FileNamelndices] 
        FileNameList = self.buildCSVFileNameList(top = PathName) 
        #print FileNameList 
        for FileName in FileNameList: 
            Diet = self.addDictWordsFromFile(PathName+FileName, Dict, TF1DFThreshold, TFIDFTTopping, TFRankCutoff) 
            print 'the KB dictionary is \n' #print Dict WordListSorted = sorted(list(Dict)) print WordListSorted WordCorpusDataFrame = pd.DataFrame(WordListSorted, columns = ['KeyWord']) print WordCorpusDataFrame 
            filePath = self.KBKeyWordCorpusPath 
            WordCorpusDataFrame.to_csv(filePath + self.SelectedKBWordCorpusFileName, encoding ='utf8') self.verificationWordCorpusConsistency(Dict, filePath + self.SelectedKBWordCorpusFileName) 

    def verificationWordCorpusConsistency(self, Dict, fileName): 
        WC = pd.read_csv(fileName, encoding ='utf8') 
        print set(WC[KeyWord'])-Dict print print Dict set(WCrKeyWord'll 
        print 'size of original diet is', 
        len(Dict) 
        print 'size of reloaded diet is ', len(set(WC['KeyWord'])) 
        print 'note inconsistency found' 

    def attachStringToCSVFileName(self, inputfilename, addstring): namelist = inputfilename.split('/') 


    def findFileName(self, RankedTF_dataframe): 
        nameList = [word for word in RankedTF_dataframe['Word'] if word[0:1] == 'kb'] 
        return ''.join(nameList) 

    def plotAIIKBFilteredResults(self): 
        PathName = self.KBArticleTFIDFDataPath 
        #PathName2 = self.KBArticleTFIDFDataFilteredPath 
        FileNameList = self.buildCSVFileNameList(top = PathName) 
        #pldfs = pldf.pldataframe()
        
        #print FileNameList 
        for FileName in FileNameList: 
            RankedTF_dataframe = pd.read_csv(PathName+FileName, encoding ='utf8') 
            DataFrameFiltered = pd.read_csv(self.attachStringToCSVFileName(FileName, Lftr', encoding ='utf8') 
            savefigName = self.KBArticleTFIDFDataFilteredPath + FileName[:-4]-F'.pdf' 
            #pldfs.plotSingleKBArticleFilteredDataFrame(RankedTF_dataframe, DataFrameFiltered, savefigName) 
            print FileName 
            
    def SingleKBArticleDataIntegration(self, SingleArticleTFIDFRDD_collect, CorpusHash TFRankDict, SingleArticleHashWordDict, SingleArticleHash TFDict, saveFigName, saveDataName): 
        SingleArticle_RankedTFIDF_dataframe = self.SparseVector2DataFrame(InputSparseVector = SingleArticleTFIDFRDD_collect[0], columnNames4HashIndexl,TFIDF,'TFIDFRankl) 
        SingleArticle RankedTFIDFAataframe['TFRankt'] = np.arrayaCorpusHash_TFRankDict[hashIndex] for hashlndex in SingleArticle_RankedTFIDF_dataframerHashIndexl) 
        SingleArticle_RankedTFIDF_dataframe[TFinArticlel = aSingleArticleHash_TFDict[hashIndex] for hashlndex in SingleArticle RankedTFIDF_dataframerHashIndexT) 
        SingleArticle 
        RankedTFIDF 
        dataframe[Word] = ([SingleArticleHashWordDict[hashlndex] for hashlndex in SingleArticle_RankedTFIDF_dataframerHashIndexl]) 
        #pldfs = pldf.pldataframe() 
        print 'Single KB article ranked TFIDF data frame:' pd.set option('display.width', 1000) print( SingleArticle_RankedTFIDF_dataframe) 
        RankedTF_dataframe = SingleArticle_RankedTFIDF_dataframe.sort(columns=TFRankl, ascending=True) 