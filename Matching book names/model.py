# imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# input file path for  testing (change this for testing)
file_path = './sample.txt'

# Stop words
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
              'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
              'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
              'itself', 'they', 'them', 'their', 'theirs', 'themselves',
              'what', 'which', 'who', 'whom', 'this', 'that', 'these',
              'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
              'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
              'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
              'against', 'between', 'into', 'through', 'during', 'before',
              'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
              'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
              'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
              'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
              'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now']

# open the input file and fetch the data
with open(file_path, encoding='utf-8') as input_file:
    N = next(input_file)
    # separate queries and results
    queries, search_results = input_file.read().split('\n*****\n')

# splits queries and results line by line to form a list
queries = queries.split('\n')
search_results = search_results.split('\n')

# Initialise TF-IDF vectorizer and fit on search results
vectorizer = TfidfVectorizer(stop_words=stop_words)
vectorizer.fit(search_results)
# transform queries and results
search_results = vectorizer.transform(search_results)
queries = vectorizer.transform(queries)

# list for cosine similarity
cos_sim = []

# find cosine similarity of each search result to all queries(documents)
for result in search_results:
    cos_sim.append(linear_kernel(result, queries).flatten())

# print the most similar query's index values (note: index starts from 1)
for similarities in cos_sim:
    print(similarities.argsort()[-1] + 1)
