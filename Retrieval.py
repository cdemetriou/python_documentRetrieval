# -*- coding: utf-8 -*-
"""\
--------------------------------------------------------------------------------
USE: python <PROGNAME> (options) file1...fileN
OPTIONS:
    -h : print this help message
    -s FILE : use stoplist file FILE
    -c FILE : name the documents file FILE
    -d FILE : name the file to store the index FILE
            (not specifying output file will store in default file "index.txt")
    -p : option to disable stemming of words (must be provided at both indexing
         and retrieval to work properly. Stemming is applied by default)
    -r FILE : name the file to store the retrieved documents FILE
            (not specifying output file will store in default file "results.txt")
    -i FILE : use existing index file FILE
    -B : enable boolean retrieval instead of ranked
    -T : use term frequencies as weights instead of TFIDF
    -q  ID  : name the query id number (from queries.txt) 
    -q FILE : or name the text file containing a set of queries
    -q "QUERY" : or a custom query in double quotation marks
    
--------------------------------------------------------------------------------
"""

"""  IMPORTS  """
import  sys, re, getopt
from read_documents import ReadDocuments
from nltk.stem import PorterStemmer
import ast # Abstract Syntax Trees
import string
import math

"""
================================================================================
    METHODS
================================================================================
"""
# METHOD SHOWING THE AVAILABLE OPTIONS FOR THE PROGRAM
def printHelp():
    
    help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
    print(help, file=sys.stderr)
    sys.exit()

    
# METHOD READING IN THE STOP WORDS    
def readStopList(file):
   
    f = open(file,'r')
    for line in f:
        stops.add(line.strip())
    
        
# METHOD USED TO PROCESS DOCUMENTS AND QUERIES LINES INTO TOKENS
def PreProcess(text, stopwords):
   
    temp_text = []
    stemmer = PorterStemmer()
    
    # Regex used to split words like: listArray -> list Array
    regex_lowerUpper = re.compile(r'(?<=[a-z])(?=[A-Z])')
    text = regex_lowerUpper.sub(' ', text)
   
    # Regex used to remove punctuation
    regex_puncts = re.compile('[%s]' % re.escape(string.punctuation))
    text  = regex_puncts.sub(' ', text)
   
    # Tokenise alphabetical sequences
    text = re.compile(r'[A-Za-z]+').findall(text) 
    
    # Lowercase, remove stopwords, stemming
    for word in text:
        word = word.lower() 
        if word not in stopwords:
            # Will not use stemming if user disables it
            if stemming:
                word = stemmer.stem(word)
            temp_text.append(word)
            
    return temp_text
 
    
# METHOD STORING THE INVERTED INDEX ALONG WITH THE IDF OF EACH WRORD AND THE 
# SIZE OF EACH DOCUMENT VECTOR IN THE GIVEN OR DEFAULT TEXT FILE
def StoreIndex(dictionary, wordsIdf, docSize, indexFile):
    
    if indexFile == "":
        indexFile = "index.txt"
    # Store index and values calculated from index in txt file per line
    with open(indexFile, "w") as f: 
        f.write("%s\n" % (dictionary)) # first line
        f.write("%s\n" % (wordsIdf))  # second line      
        f.write("%s\n" % (docSize)) # third line
   
 
# METHOD CREATING THE INVERTED INDEX AND CALCULATING THE IDF OF EACH WRORD AND  
# THE SIZE OF EACH DOCUMENT VECTOR
def CreateIndex(file,stopwords, indexFile):
    
    dictionary = {} 
    wordsIdf = {} # Importance of each word in each document
    totalDocs = 0
    docSize = {}
    dfs = {}

    # Inverted index population
    for doc in file:
        # Count number of documents
        totalDocs += 1
        for line in doc.lines:
            words = PreProcess(line, stopwords) 
            # create index dictionary and populate it with each word in each 
            # document and its TF - Term Freq of word in each doc
            for word in words: 
                if word not in dictionary:
                    dictionary[word] = dict()
                    dictionary[word][doc.docid] = 1 
                else:
                    if doc.docid not in dictionary[word]:
                        dictionary[word][doc.docid] = 1 
                    else:
                        dictionary[word][doc.docid] += 1 
       
    # DF - In how many docs does a word appear in
    # Calculate IDF = log(|D| / dfw) - importance of word in the collection
    for word in dictionary:
        dfs[word] = len(dictionary[word])
        wordsIdf[word] = math.log10(totalDocs / dfs[word])
       
    # THE DOCUMENT VECTOR SIZE - The sum of squared weights 
    # for terms appearing in each document - Σ((TFwd.IDFw)**2)
    for word in dictionary:
        for doc.docid in dictionary[word]: 
            if doc.docid not in docSize:
                docSize[doc.docid] = math.pow((int(dictionary[word][doc.docid])\
                                             * float(wordsIdf[word])),2)
            else:
                docSize[doc.docid] += math.pow((int(dictionary[word][doc.docid])\
                                             * float(wordsIdf[word])),2)
    
    StoreIndex(dictionary, wordsIdf, docSize, indexFile)
  
    
# METHOD READING IN THE INVERTED INDEX ALONG WITH THE IDF OF EACH WRORD AND THE 
# SIZE OF EACH DOCUMENT VECTOR FROM THE GIVEN TEXT FILE
def ReadFromFile(file):
   
    index = {}
    wordsIdf = {}
    docSize = {}
    
    with open(file, "r") as file:
        # Retrieve elements per line
        # literal_eval - Safely evaluate a string containing a Python literal
        for line_num, line in enumerate(file):
            if line_num == 0:
                index = ast.literal_eval(line)
            elif line_num == 1:
                wordsIdf = ast.literal_eval(line)
            elif line_num == 2:
                 docSize = ast.literal_eval(line)
            elif line_num > 2:
                break    
  
    return index, wordsIdf, docSize

    
# METHOD USING THE PROVIDED QUERY OR QUERIES FILE TO PROCESS ITS WORDS
def GetQuery(given_query, stopswords, index, wordsIdf, docSize, resultsFile):
  
    wordsFreq = {}
    querySize = 0 
    
    # Condition entered when user provides a query id    
    if len(given_query) < 3:
        qID = int(given_query)
        queries = ReadDocuments("queries.txt")
        for doc in queries:
            docID = int(doc.docid) 
            if docID == qID:
                for line in doc.lines:
                    words = PreProcess(line, stopswords)
                    for word in words: 
                        # TFw - Term Freq of each word in query
                        if word not in wordsFreq:
                            wordsFreq[word] = 1
                        else:
                            wordsFreq[word] += 1

                # THE QUERY VECTOR SIZE - The sum of squared weights 
                # for terms appearing in each document  
                for word in wordsFreq:
                    querySize += math.pow(wordsFreq[word],2)
                    
                findSIMandPrint(qID, wordsFreq, querySize, index,wordsIdf, \
                                docSize,resultsFile)
                break # Exit loop when the query is found and analyzed
    
                
    # Condition entered when the user provides text file to read queries from             
    elif given_query.endswith(".txt"): 
        queries = ReadDocuments(given_query)
        for doc in queries:
            wordsFreq = {}
            qID = int(doc.docid)
            querySize = 0
            for line in doc.lines:
                    words = PreProcess(line, stopswords)
                    for word in words: 
                        # TFw - Term Freq of each word in query
                        if word not in wordsFreq:
                            wordsFreq[word] = 1
                        else:
                            wordsFreq[word] += 1

            # THE QUERY VECTOR SIZE - The sum of squared weights 
            # for terms appearing in each document  
            for word in wordsFreq:
                querySize += math.pow(wordsFreq[word],2)
                
            findSIMandPrint(qID, wordsFreq, querySize, index, wordsIdf, \
                            docSize, resultsFile)
    
    # Condition entered when the user provides a custom query
    else:
        query = str(given_query)
        qID = 1
        words = PreProcess(query, stopswords)
        for word in words: 
            # TFw - Term Freq of each word in query
            if word not in wordsFreq:
                wordsFreq[word] = 1
            else:
                wordsFreq[word] += 1 

        # THE QUERY VECTOR SIZE - The sum of squared weights 
        # for terms appearing in each document  
        for word in wordsFreq:
            querySize += math.pow(wordsFreq[word],2)
            
        findSIMandPrint(qID, wordsFreq, querySize, index, wordsIdf, docSize, \
                        resultsFile)


# METHOD COMPUTING THE QUERY - DOCUMENT SIMILARITY, SORTING THE RESULTS AND 
# STORING THEM TO THE GIVEN OR DEFAULT FILE        
def findSIMandPrint(qID, wordsFreq, querySize, index, wordsIdf, docSize, resultsFile):      
    
    sim = {}
    
    # Retrieve documents based on ranking
    if booleanRetrieval == False:
        ## Calculating the sum of qwi * dwi - Σ (TFqw*IDFw  *  TFdw*IDFw)
        for word in wordsFreq:
            if word in index:
                for docid in index[word]:
                    if docid not in sim:    
                        sim[docid] = (int(wordsFreq[word])*float(wordsIdf[word]))*\
                                     (int(index[word][docid])*float(wordsIdf[word]))
                    else:
                        sim[docid] += (int(wordsFreq[word])*float(wordsIdf[word]))*\
                                     (int(index[word][docid])*float(wordsIdf[word]))
        
        # Divide the sum of weights by the query vector size * document vector size
        for docid in sim:
            sim[docid] = sim[docid] / \
                        (math.sqrt(querySize) * math.sqrt(docSize[docid]))
               
        # Sorting the similarity dictionary descending by value 
        simDocs = sorted(sim, reverse=True, key=lambda docID: sim[docID])
       
        # Write in the given or default text file the first 10 retrieved documents
        # for each query
        count = 0
        with open(resultsFile, "a") as f:
            for docID in simDocs:
                f.write("%i %i\n" % (qID,docID))
                count = count +1
                if count == 10:
                    break
            
    # Retrieve documents based on all query words matching in a document
    elif booleanRetrieval:
        count = 0
        wordMatch = {}
        # Count number of matching query words in each document 
        querywordsCount = 0
        for word in wordsFreq:
            querywordsCount += 1
            if word in index:
                for docid in index[word]:
                    if docid not in sim:    
                        wordMatch[docid] = 1
                    else:
                        wordMatch[docid] += 1

        for docID in wordMatch:
            if wordMatch[docID] == querywordsCount:
                with open(resultsFile, "a") as f:
                    f.write("%i %i\n" % (qID,docID))
                    count = count +1
                    if count == 10:
                        break
                    
"""
================================================================================
    MAIN PROGRAM
================================================================================
"""  

# Try block to catch any unrecognized option entries or wrongly provided options
try:  
    opts, args = getopt.getopt(sys.argv[1:],'hs:d:pr:c:i:BTq:')
    opts = dict(opts)
except getopt.GetoptError as err:
    # May print "option -a not recognized" or "option -q requires argument"
    print("\nERROR OCCURED:  "+str(err)) 
    printHelp()
    sys.exit()
    
stops = set()
documents = set()
indexFile = ""
resultsFile = ""
stemming = True
booleanRetrieval = False

# Help option
if '-h' in opts: 
    printHelp()

# option for providing Stopwords file    
if '-s' in opts:
    readStopList(opts['-s'])

# option for providing output index file    
if '-d' in opts:
    indexFile = opts['-d']

# option for disabling stemming    
if '-p' in opts:
    stemming = False
                 
# option for providing output results file    
if '-r' in opts:
    resultsFile = opts['-r']

# option for providing a collection to be indexed
if '-c' in opts:
    documents = ReadDocuments(opts['-c'])
    CreateIndex(documents,stops, indexFile)

# option to use desired preexisting inverted index file
if '-i' in opts:
    index, wordsIdf, docSize = ReadFromFile(opts['-i'])
   
# option to exersice boolean retrieval instead of ranked retrieval
try:   
    if '-B' in opts:
        booleanRetrieval = True
except NameError as err:
    print("\nERROR OCCURED: -B only used with document retrieval, not indexing") 
    printHelp()
    sys.exit() 
    
# option to use term frequencies as weights instead of TFIDF 
# change all idf values to 1 
try:   
    if '-T' in opts:
        for word_idf in wordsIdf:
            wordsIdf[word_idf] = 1
except NameError as err:
    print("\nERROR OCCURED: -T only used with document retrieval, not indexing") 
    printHelp()
    sys.exit()   
    
# option to supply query id or custom query
if '-q' in opts:
    # Initialize "results.txt" if a file name is not provided
    if resultsFile == "":
        resultsFile = "results.txt"
    with open(resultsFile, "w") as f:
        f.write("")
    GetQuery(opts['-q'], stops, index, wordsIdf, docSize, resultsFile)

    