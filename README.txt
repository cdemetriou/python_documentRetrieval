
Document retrieval system, based on the vector space model, and evaluating its performance over the CACM test collection under alternative conﬁgurations, arising from choices that might include the following: 
• stoplist: whether a stoplist is used or not (to exclude less useful terms)
• stemming: whether or not stemming is applied to terms. 
• term weighting: whether term weights in vectors are binary, or are term frequencies, or use the TF.IDF approach

----------------------------------------------------------------------------------------------------------------------------------
The following instructions show the intended use of the Retrieval.py

1. Include the following files in the same directory as the Retrieval.py
	read_documents.py //Class provided and used for reading in the documents and queries 
	documents.txt //Documents used for the index
	queries.txt //Queries used to produce results for evaluating the system
	stop_list.txt //Frequent words to be excluded
	
2. Though the command line change directory to the folder containing the above files
3. The Retrieval.py is run as shown below:
	python Retrieval.py
followed by command line options for specifying input/output files and functionalities
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

First run:
	python Retrieval.py -s stop_list.txt -c documents.txt
to create the index from the supplied documents and inlude removal of stopwords

Then run:
	python Retrieval.py -i index.txt -s stop_list.txt -q queries.txt -r retrieved.txt
to use the previously created index to retrieve relevant documents for the provided queries
use stopwords on query terms and specify retrieved.txt as results file

The results produced from the above code are by default stored in the results.txt file 
unless a file name is given as shown above.
