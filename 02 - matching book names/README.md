We are given a set of book names and a set of descriptions (about the given books) but they are not in order.
Our model will take the description and map it to the correct book name.

For this purpose we make use of tf-idf representation of words and predict cosine similarity for each query (book name) and description.
