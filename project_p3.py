# CS421: Natural Language Processing
# University of Illinois at Chicago
# Fall 2023
# Project Part 3
#
# Do not rename/delete any functions or global variables provided in this template and write your solution
# in the specified sections. Use the main function to test your code when running it from a terminal.
# Avoid writing that code in the global scope; however, you should write additional functions/classes
# as needed in the global scope. These templates may also contain important information and/or examples
# in comments so please read them carefully. If you want to use external packages (not specified in
# the assignment) then you need prior approval from course staff.
# =========================================================================================================

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import string
import re
import csv
import nltk

#-----------------------------------CODE FROM PART 1--------------------------------------------------

# Before running code that makes use of Word2Vec, you will need to download the provided w2v.pkl file
# which contains the pre-trained word2vec representations from Blackboard
#
# If you store the downloaded .pkl file in the same directory as this Python
# file, leave the global EMBEDDING_FILE variable below as is.  If you store the
# file elsewhere, you will need to update the file path accordingly.
EMBEDDING_FILE = "w2v.pkl"


# Function: load_w2v
# filepath: path of w2v.pkl
# Returns: A dictionary containing words as keys and pre-trained word2vec representations as numpy arrays of shape (300,)
def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)


# Function: load_as_list(fname)
# fname: A string indicating a filename
# Returns: Two lists: one a list of document strings, and the other a list of integers
#
# This helper function reads in the specified, specially-formatted CSV file
# and returns a list of documents (documents) and a list of binary values (label).
def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels


# Function: extract_user_info(user_input)
# user_input: A string of arbitrary length
# Returns: name as string
def extract_user_info(user_input):
    name = ""
    name_match = re.search(r"(^|\s)([A-Z][A-Za-z-&'\.]*(\s|$)){2,4}", user_input)
    if name_match is not None:
        name = name_match.group(0).strip()
    return name


# Function to convert a given string into a list of tokens
# Args:
#   inp_str: input string
# Returns: token list, dtype: list of strings
def get_tokens(inp_str):
    return inp_str.split()


# Function: vectorize_train, see project statement for more details
# training_documents: A list of strings
# Returns: An initialized TfidfVectorizer model, and a document-term matrix, dtype: scipy.sparse.csr.csr_matrix
def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer(tokenizer=get_tokens, lowercase=True)

    # Compute TF-IDF representations
    tfidf_train = vectorizer.fit_transform(training_documents)


    return vectorizer, tfidf_train


# Function: w2v(word2vec, token)
# word2vec: The pretrained Word2Vec representations as dictionary
# token: A string containing a single token
# Returns: The Word2Vec embedding for that token, as a numpy array of size (300,)
#
# This function provides access to 300-dimensional Word2Vec representations
# pretrained on Google News.  If the specified token does not exist in the
# pretrained model, it should return a zero vector; otherwise, it returns the
# corresponding word vector from the word2vec dictionary.
def w2v(word2vec, token):
    word_vector = np.zeros(300, )  # Initialize a zero vector

    if token in word2vec:
        word_vector = word2vec[token]


    return word_vector


# Function: string2vec(word2vec, user_input)
# word2vec: The pretrained Word2Vec model
# user_input: A string of arbitrary length
# Returns: A 300-dimensional averaged Word2Vec embedding for that string
#
# This function preprocesses the input string, tokenizes it using get_tokens, extracts a word embedding for
# each token in the string, and averages across those embeddings to produce a
# single, averaged embedding for the entire input.
def string2vec(word2vec, user_input):
    tokens = get_tokens(user_input)
    token_embeddings = [w2v(word2vec, token) for token in tokens]

    # Average the token embeddings
    if len(token_embeddings) > 0:
        averaged_embedding = np.mean(token_embeddings, axis=0)
    else:
        averaged_embedding = np.zeros(300, )  # Return a zero vector if no tokens are found

    return averaged_embedding


# Function: instantiate_models()
# This function does not take any input
# Returns: Four instantiated machine learning models
#
# This function instantiates the four imported machine learning models, and
# returns them for later downstream use.  You do not need to train the models
# in this function.
def instantiate_models():
    nb = GaussianNB()
    logistic = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)
    mlp = MLPClassifier(random_state=100)

    return nb, logistic, svm, mlp


# Function: train_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# tfidf_train: A document-term matrix built from the training data
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model_tfidf(model, tfidf_train, training_labels):
    # Fit the model on TF-IDF features
    model.fit(tfidf_train.toarray(), training_labels)

    return model



# Function: train_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# training_data: A list of training documents
# training_labels: A list of integers (all 0 or 1)
# Returns: A trained version of the input model
#
# This function trains an input machine learning model using averaged Word2Vec
# embeddings for the training documents.
def train_model_w2v(model, word2vec, training_documents, training_labels):
    # Convert training documents into averaged Word2Vec embeddings
    training_embeddings = [string2vec(word2vec, doc) for doc in training_documents]

    # Convert the list of embeddings to a NumPy array
    training_embeddings_array = np.array(training_embeddings)

    # Fit the model on Word2Vec embeddings
    model.fit(training_embeddings_array, training_labels)

    return model


# Function: test_model_tfidf(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# vectorizer: An initialized TfidfVectorizer model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_tfidf(model, vectorizer, test_documents, test_labels):
    # Extract TF-IDF vector using the vectorizer
    tfidf_test = vectorizer.transform(test_documents)

    # Convert TF-IDF vector to dense array
    tfidf_test_array = tfidf_test.toarray()

    # Predict labels
    predicted_labels = model.predict(tfidf_test_array)

    # Compute evaluation metrics
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    accuracy = accuracy_score(test_labels, predicted_labels)

    return precision, recall, f1, accuracy


# Function: test_model_w2v(model, word2vec, training_documents, training_labels)
# model: An instantiated machine learning model
# word2vec: A pretrained Word2Vec model
# test_data: A list of test documents
# test_labels: A list of integers (all 0 or 1)
# Returns: Precision, recall, F1, and accuracy values for the test data
#
# This function tests an input machine learning model by extracting features
# for each preprocessed test document and then predicting an output label for
# that document.  It compares the predicted and actual test labels and returns
# precision, recall, f1, and accuracy scores.
def test_model_w2v(model, word2vec, test_documents, test_labels):
    # Convert test documents into averaged Word2Vec embeddings
    test_embeddings = [string2vec(word2vec, doc) for doc in test_documents]

    # Convert the list of embeddings to a NumPy array
    test_embeddings_array = np.array(test_embeddings)

    # Predict labels
    predicted_labels = model.predict(test_embeddings_array)

    # Compute evaluation metrics
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    accuracy = accuracy_score(test_labels, predicted_labels)

    return precision, recall, f1, accuracy

#-----------------------------------CODE FROM PART 2--------------------------------------------------

# Function: count_words(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
# Implement the count_words function in project_p2.py. Given a raw user
# input string, use the nltk.tokenize.word tokenize function to break down
# the input into tokens. Then, filter all punctuation tokens from the string using string.punctuation.
# 3 Finally, return the remaining number of tokens or words.
# This function counts the number of words in the input string.
def count_words(user_input):
    #num_words = 0
    # Tokenize the input
    tokens = nltk.word_tokenize(user_input)

    # Filter out punctuation tokens
    tokens = [token for token in tokens if token not in string.punctuation]

    # Count the remaining tokens (words)
    num_words = len(tokens)
    # [YOUR CODE HERE]


    return num_words

# Function: words_per_sentence(user_input)
# user_input: A string of arbitrary length
# Returns: A floating point value
# Implement the words_per_sentence function in project_p2.py. Given a raw
# user input string, use the nltk.tokenize.sent_tokenize function to first segment the input into sentences. Then use the count_words function created for
# Q1 to count the number of words in each sentence. Take the sum of these counts
# and divide it by the total number of sentences in the input to return the average.
# This function computes the average number of words per sentence
def words_per_sentence(user_input):
    #wps = 0.0
    # [YOUR CODE HERE]
    # Tokenize the input into sentences
    # Split the input into sentences based on periods (.)
    sentences = nltk.sent_tokenize(user_input)

    total_word_count = 0
    total_sentence_count = len(sentences)

    for sentence in sentences:
        word_count = count_words(sentence)
        total_word_count += word_count

    if total_sentence_count > 0:
        wps = total_word_count / total_sentence_count
    else:
        wps = 0.0
    return wps


# Function: get_pos_tags(user_input)
# user_input: A string of arbitrary length
# Returns: A list of (token, POS) tuples
# Implement the get_pos_tags function in project_p2.py. Given a raw user
# input string, the function should return a list of tuples in the format (token,
# POS), using part-of-speech (POS) tags from the Penn Treebank tagset. Use the
# NLTK word_tokenize function to tokenize the input (you can ignore sentence
# boundaries for simplicity). You can use the nltk.pos_tag function to compute
# POS tags for the tokenized text.
# This function tags each token in the user_input with a Part of Speech (POS) tag from Penn Treebank.
def get_pos_tags(user_input):
    # tagged_input = []
    # Tokenize the input
    tokens = nltk.word_tokenize(user_input)

    # Get POS tags for the tokens
    tagged_input = nltk.pos_tag(tokens)
    # [YOUR CODE HERE]


    return tagged_input


# Function: get_pos_categories(tagged_input)
# tagged_input: A list of (token, POS) tuples
# Returns: Seven integers, corresponding to the number of pronouns, personal
#          pronouns, articles, past tense verbs, future tense verbs,
#          prepositions, and negations in the tagged input
# Implement the get_pos_categories function in project_p2.py. Refer to Table 2 to see which Penn Treebank POS tags belong to each category. The
# function should follow these guidelines when counting the number of tokens belonging to different categories. For example, if your input has five tokens with
# POS tags of VBD and three tokens with POS tags of VBN, then your input has
# a total of eight past tense verbs.
# Linguistic Feature         Penn Treebank Tag(s)
# Pronouns                   PRP, PRP$, WP, WP$
# Personal pronouns          PRP
# Articles                   DT
# Past tense verbs           VBD, VBN
# Future tense verbs         MD
# Prepositions               IN
# Table 2: Penn Treebank POS tags belonging to different linguistic feature categories. Note that in addition to articles, indefinite determiners and some instances of “all” and “both” are also labeled as DT under the Penn Treebank
# labeling scheme; it’s fine to go ahead and keep anything labeled as DT in your
# count.
# This function counts the number of tokens corresponding to each of six POS tag
# groups, and returns those values.  The Penn Treebag tags corresponding that
# belong to each category can be found in Table 2 of the project statement.
def get_pos_categories(tagged_input):
    num_pronouns = 0
    num_prp = 0
    num_articles = 0
    num_past = 0
    num_future = 0
    num_prep = 0

    for token, pos in tagged_input:
        if pos in ["PRP", "PRP$", "WP", "WP$"]:
            num_pronouns += 1
        if pos == "PRP":
            num_prp += 1
        if pos == "DT":
            num_articles += 1
        if pos in ["VBD", "VBN"]:
            num_past += 1
        if pos == "MD":
            num_future += 1
        if pos == "IN":
            num_prep += 1
    # [YOUR CODE HERE]


    return num_pronouns, num_prp, num_articles, num_past, num_future, num_prep


# Function: count_negations(user_input)
# user_input: A string of arbitrary length
# Returns: An integer value
# Implement the count_negations function in project_p2.py. The function
# should count the number of negation terms present in the raw user input. For
# the sake of simplicity, restrict your set of negation terms to {no, not, never,
# n’t}. For example, the sentence “I haven’t finished the project” contains one
# negation. You can find a more comprehensive list of negations here: https:
# //www.grammarly.com/blog/negatives/.
# This function counts the number of negation terms in a user input string
def count_negations(user_input):
    #num_negations = 0
    negation_terms = [
        "no",
        "not",
        "never",
        "n't",
        ]

    # Tokenize the input
    tokens = nltk.word_tokenize(user_input)

    # Count the number of negation terms in the input
    num_negations = sum(1 for token in tokens if token.lower() in negation_terms)
    # [YOUR CODE HERE]


    return num_negations
# Function: summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations)
# num_words: An integer value
# wps: A floating point value
# num_pronouns: An integer value
# num_prp: An integer value
# num_articles: An integer value
# num_past: An integer value
# num_future: An integer value
# num_prep: An integer value
# num_negations: An integer value
# Returns: A list of three strings
#
# This function identifies the three most informative linguistic features from
# among the input feature values, and returns the psychological correlates for
# those features.  num_words and/or wps should be included if, and only if,
# their values exceed predetermined thresholds.  The remainder of the three
# most informative features should be filled by the highest-frequency features
# from among num_pronouns, num_prp, num_articles, num_past, num_future,
# num_prep, and num_negations.
# At this point, you have computed numerous informative features describing the
# user’s input: the word count, the number of words per sentence, the number of
# pronouns, the number of personal pronouns, the number of articles, the number
# of past tense verbs, the number of future tense verbs, the number of prepositions,
# and the number of negations. Implement the summarize_analysis function
# in project_p2.py to identify the most informative of these characteristics as
# follows.
# First, set threshold values for word count and words per sentence such that
# these features are only considered highly informative if they exceed that threshold. Although you may ultimately want to experiment with multiple thresholds
# in the final version of your chatbot to figure out what works best for your specific
# model, in this part of the project you can set the word count threshold to 100
# and the words per sentence threshold to 20. These thresholds were computed
# using simple heuristics: most English-language sentences are 20 words or less,4
# and most English-language paragraphs are five sentences or less.5
# Check to see if the computed word count and/or words per sentence exceed the predetermined thresholds. If they do, add their psychological correlates (see Table 1 or the skeleton code for details) to your list of informative
# correlates. Then, order the remaining linguistic features (pronouns, personal
# pronouns, articles, past tense verbs, future tense verbs, prepositions, and negations) by frequency from highest to lowest, and add psychological correlates in
# that order until your list of informative correlates has reached a size of 3. Ties
# can be broken in the order presented in Table 1. For instance, if num_prp and
# num_pronouns is the same and your list only has room for one correlate, then you
# would pick the num_pronouns correlate to include in your list. More concretely,
# if the following feature values were passed into summarize_analysis():
# num_words = 50
# wps = 26.2
# num_pronouns = 6
# num_prp = 4
# num_articles = 3
# num_past = 1
# num_future = 0
# num_prep = 5
# num_negations = 5
# You might return: “Verbal fluency, cognitive complexity,” “Informal, personal,” and “Education, concern with precision.”
def summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations):
    informative_correlates = []

    # Creating a reference dictionary with keys = linguistic features, and values = psychological correlates.
    # informative_correlates should hold a subset of three values from this dictionary.
    # DO NOT change these values for autograder to work correctly
    psychological_correlates = {}
    psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
    psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
    psychological_correlates["num_pronouns"] = "Informal, personal"
    psychological_correlates["num_prp"] = "Personal, social"
    psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
    psychological_correlates["num_past"] = "Focused on the past"
    psychological_correlates["num_future"] = "Future and goal-oriented"
    psychological_correlates["num_prep"] = "Education, concern with precision"
    psychological_correlates["num_negations"] = "Inhibition"

    # Set thresholds
    num_words_threshold = 100
    wps_threshold = 20

    # [YOUR CODE HERE]
    # Check if num_words and wps exceed thresholds
    if num_words > num_words_threshold:
        informative_correlates.append(psychological_correlates["num_words"])
    if wps > wps_threshold:
        informative_correlates.append(psychological_correlates["wps"])

    # Sort remaining linguistic features by frequency
    feature_counts = {
        "num_pronouns": num_pronouns,
        "num_prp": num_prp,
        "num_articles": num_articles,
        "num_past": num_past,
        "num_future": num_future,
        "num_prep": num_prep,
        "num_negations": num_negations,
    }

    # Sort features by frequency in descending order
    sorted_features = sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)

    # Add the most frequent features to informative_correlates until it has 3 elements
    for feature, count in sorted_features:
        if len(informative_correlates) < 3:
            informative_correlates.append(psychological_correlates[feature])
        else:
            break

    return informative_correlates
#-----------------------------------NEW IN PART 3--------------------------------------------------

# Function: welcome_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements the chatbot's welcome states.  Feel free to customize
# the welcome message!  In this state, the chatbot greets the user.
def welcome_state():
    # Display a welcome message to the user
    user_input = print("Welcome to the CS 421 chatbot!  ")

    return "get_user_info"


# Function: get_info_state()
# This function does not take any input
# Returns: A string indicating the next state and a string indicating the
#          user's name
#
# This function implements a state that requests the user's name and then processes
# the user's response to extract that information.
def get_info_state():
    # Request the user's name, and accept a user response of
    # arbitrary length.  Feel free to customize this!
    user_input = input("What is your name?\n")

    # Extract the user's name
    name = extract_user_info(user_input)

    return "sentiment_analysis", name


# Function: sentiment_analysis_state(name, model, vectorizer, word2vec)
# name: A string indicating the user's name
# model: The trained classification model used for predicting sentiment
# vectorizer: OPTIONAL; The trained vectorizer, if using TFIDF (leave empty otherwise)
# word2vec: OPTIONAL; The pretrained Word2Vec model, if using Word2Vec (leave empty otherwise)
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what they want to talk about,
# and then processes their response to predict their current sentiment.
def sentiment_analysis_state(name, model, vectorizer=None, word2vec=None):
    # Check the user's sentiment
    user_input = input("Thanks {0}!  What do you want to talk about today?\n".format(name))

    # Predict the user's sentiment
    # test = vectorizer.transform([user_input])  # Use if you selected a TFIDF model
    test = string2vec(word2vec, user_input)  # Use if you selected a w2v model

    label = None
    label = model.predict(test.reshape(1, -1))

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))

    return "stylistic_analysis"


# Function: stylistic_analysis_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that asks the user what's on their mind, and
# then analyzes their response to identify informative psycholinguistic correlates.
def stylistic_analysis_state():
    user_input = input("I'd also like to do a quick stylistic analysis. What's on your mind today?\n")
    num_words = count_words(user_input)
    wps = words_per_sentence(user_input)
    pos_tags = get_pos_tags(user_input)
    num_pronouns, num_prp, num_articles, num_past, num_future, num_prep = get_pos_categories(pos_tags)
    num_negations = count_negations(user_input)

    # Uncomment the code below to view your output from each individual function
    # print("num_words:\t{0}\nwps:\t{1}\npos_tags:\t{2}\nnum_pronouns:\t{3}\nnum_prp:\t{4}"
    #      "\nnum_articles:\t{5}\nnum_past:\t{6}\nnum_future:\t{7}\nnum_prep:\t{8}\nnum_negations:\t{9}".format(
    #    num_words, wps, pos_tags, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations))

    # Generate a stylistic analysis of the user's input
    informative_correlates = summarize_analysis(num_words, wps, num_pronouns,
                                                num_prp, num_articles, num_past,
                                                num_future, num_prep, num_negations)
    print(
        "Thanks!  Based on my stylistic analysis, I've identified the following psychological correlates in your response:")
    for correlate in informative_correlates:
        print("- {0}".format(correlate))

    return "check_next_action"


# Function: check_next_state()
# This function does not take any input
# Returns: A string indicating the next state
#
# This function implements a state that checks to see what the user would like
# to do next.  The user should be able to indicate that they would like to quit
# (in which case the state should be "quit"), redo the sentiment analysis
# ("sentiment_analysis"), or redo the stylistic analysis
# ("stylistic_analysis").
def check_next_state():
    next_state = ""

    # Write your code here:
    print("What would you like to do next?")
    print("(a) Quit the conversation")
    print("(b) Redo the sentiment analysis")
    print("(c) Redo the stylistic analysis")

    user_choice = input("Enter your choice: ")

    if user_choice.lower() == 'a':
        next_state = "quit"
    elif user_choice.lower() == 'b':
        next_state = "sentiment_analysis"
    elif user_choice.lower() == 'c':
        next_state = "stylistic_analysis"
    else:
        print("Sorry, I didn't understand that choice. Please enter 'a', 'b', or 'c'.")

    return next_state


# Function: run_chatbot(model, vectorizer=None):
# model: A trained classification model
# vectorizer: OPTIONAL; The trained vectorizer, if using Naive Bayes (leave empty otherwise)
# word2vec: OPTIONAL; The pretrained Word2Vec model, if using other classification options (leave empty otherwise)
# Returns: This function does not return any values
#
# This function implements the main chatbot system---it runs different
# dialogue states depending on rules governed by the internal dialogue
# management logic, with each state handling its own input/output and internal
# processing steps.  The dialogue management logic should be implemented as
# follows:
# welcome_state() (IN STATE) -> get_info_state() (OUT STATE)
# get_info_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE)
# sentiment_analysis_state() (IN STATE) -> stylistic_analysis_state() (OUT STATE - First time sentiment_analysis_state() is run)
#                                    check_next_state() (OUT STATE - Subsequent times sentiment_analysis_state() is run)
# stylistic_analysis_state() (IN STATE) -> check_next_state() (OUT STATE)
# check_next_state() (IN STATE) -> sentiment_analysis_state() (OUT STATE option 1) or
#                                  stylistic_analysis_state() (OUT STATE option 2) or
#                                  terminate chatbot
def run_chatbot(model, vectorizer=None, word2vec=None):
    # Write your code here:
    state = "welcome_state"

    while state != "quit":
        if state == "welcome_state":
            state = welcome_state()
        elif state == "get_user_info":
            state, name = get_info_state()
        elif state == "sentiment_analysis":
            state = sentiment_analysis_state(name, model, vectorizer, word2vec)
        elif state == "stylistic_analysis":
            state = stylistic_analysis_state()
        elif state == "check_next_action":
            state = check_next_state()

    return


# This is your main() function.  Use this space to try out and debug your code
# using your terminal.  The code you include in this space will not be graded.
if __name__ == "__main__":
    # Set things up ahead of time by training the TfidfVectorizer and Naive Bayes model
    documents, labels = load_as_list("dataset.csv")

    # Load the Word2Vec representations so that you can make use of it later
    word2vec = load_w2v(EMBEDDING_FILE)  # Use if you selected a Word2Vec model

    # Compute TFIDF representations so that you can make use of them later
    # vectorizer, tfidf_train = vectorize_train(documents)  # Use if you selected a TFIDF model

    # Instantiate and train the machine learning models
    # To save time, only uncomment the lines corresponding to the sentiment
    # analysis model you chose for your chatbot!

    # nb_tfidf, logistic_tfidf, svm_tfidf, mlp_tfidf = instantiate_models() # Uncomment to instantiate a TFIDF model
    nb_w2v, logistic_w2v, svm_w2v, mlp_w2v = instantiate_models()  # Uncomment to instantiate a w2v model
    # nb_tfidf = train_model_tfidf(nb_tfidf, tfidf_train, labels)
    # nb_w2v = train_model_w2v(nb_w2v, word2vec, documents, labels)
    # logistic_tfidf = train_model_tfidf(logistic_tfidf, tfidf_train, labels)
    # logistic_w2v = train_model_w2v(logistic_w2v, word2vec, documents, labels)
    # svm_tfidf = train_model_tfidf(svm_tfidf, tfidf_train, labels)
    svm_w2v = train_model_w2v(svm_w2v, word2vec, documents, labels)
    # mlp_tfidf = train_model_tfidf(mlp_tfidf, tfidf_train, labels)
    # mlp_w2v = train_model_w2v(mlp_w2v, word2vec, documents, labels)

    # ***** New in Project Part 3! *****
    # next_state = welcome_state() # Uncomment to check how this works
    # next_state, name = get_info_state() # Uncomment to check how this works
    # next_state = sentiment_analysis_state(name, svm_w2v, word2vec=word2v) # Uncomment to check how this works---note that you'll need to update parameters to use different sentiment analysis models!
    # next_state = stylistic_analysis_state() # Uncomment to check how this works
    # next_state = check_next_state() # Uncomment to check how this works

    # run_chatbot(mlp, word2vec=word2vec) # Example for running the chatbot with
                                        # MLP (make sure to comment/uncomment
                                        # properties of other functions as needed)
    run_chatbot(svm_w2v, word2vec=word2vec) # Example for running the chatbot with SVM and Word2Vec---make sure your earlier functions are copied over for this to work correctly!
