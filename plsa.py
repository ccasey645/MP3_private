import numpy as np
import math


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix


def normalize_cols(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    col_sums = input_matrix.sum(axis=0)
    new_matrix = input_matrix / row_sums[:, :, np.newaxis]
    return new_matrix


class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
        with open(self.documents_path, "r") as data_file:
            for line in data_file:
                data = line.strip().split()
                try:
                    label = int(data[0])
                except:
                    label_text_split_index = 0
                else:
                    #Todo: figure out where label goes!
                    pass
                    #label
                finally:
                    label_text_split_index = 1
                    try:
                        self.documents.append(data[label_text_split_index:])
                    except IndexError:
                        raise RuntimeError("No text data after lablel!")
                    else:
                        self.number_of_documents = len(self.documents)

    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        corpus_vocab_set = set()
        for document in self.documents:
            for word in document:
                corpus_vocab_set.add(word)

        self.vocabulary = sorted(list(corpus_vocab_set), key=lambda v: v.lower())
        self.vocabulary_size = len(self.vocabulary)


    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        
        self.term_doc_matrix = []

        for index, document in enumerate(self.documents):
            word_count = {}
            for word in document:
                try:
                    word_count[word] += 1
                except KeyError:
                    word_count[word] = 1

        for index, document in enumerate(self.documents):
            for word in self.vocabulary:
                try:
                    self.term_doc_matrix[index].append(word_count[word])
                except IndexError:
                    self.term_doc_matrix.append([])
                    try:
                        self.term_doc_matrix[index].append(word_count[word])
                    except KeyError:
                        self.term_doc_matrix[index].append(0)
                except KeyError:
                    self.term_doc_matrix[index].append(0)

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################

        self.document_topic_prob = np.random.rand(self.number_of_documents, number_of_topics)
        # for i in range(0, self.number_of_documents):
        #     self.document_topic_prob.append([])
        #     for j in range(0, number_of_topics):
        #         topic_prob = np.random.random()
        #         self.document_topic_prob[i].append(topic_prob)

        self.topic_word_prob = np.random.rand(number_of_topics, self.vocabulary_size)
        # for i in range(0, number_of_topics):
        #     self.topic_word_prob.append([])
        #     for word in self.vocabulary:
        #         vocab_prob = np.random.random()
        #         self.topic_word_prob[i].append(vocab_prob)
        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        
        # ############################
        # your code here
        # ############################
        for doc_index in range(self.number_of_documents):
            for topic_index in range(2):
                for word_index in range(self.vocabulary_size):
                    self.topic_prob[doc_index][topic_index][word_index] = self.document_topic_prob[doc_index][topic_index] * self.topic_word_prob[topic_index][word_index]
            self.topic_prob[doc_index] = normalize(self.topic_prob[doc_index].transpose()).transpose()



    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        # ############################
        for doc_index in range(self.number_of_documents):
            topic_counts = []
            for topic_index in range(number_of_topics):
                count = 0
                for word_index in range(self.vocabulary_size):
                    count += self.term_doc_matrix[doc_index][word_index] * self.topic_prob[doc_index][topic_index][word_index]
                topic_counts.append(count)
            self.document_topic_prob[doc_index, :] = normalize(np.array(topic_counts).reshape(1, -1))

        # for doc_index in range(self.number_of_documents):
        #     for topic_index in range(number_of_topics):
        #         for word_index in range(self.vocabulary_size):
        #             self.document_topic_prob[doc_index][topic_index] = self.term_doc_matrix[doc_index][word_index] * self.topic_prob[doc_index][topic_index][word_index]
        #     self.document_topic_prob = normalize(self.document_topic_prob)

        # update P(z | d)

        # ############################
        # your code here
        # ############################
        for topic_index in range(number_of_topics):
            word_counts = []
            for word_index in range(self.vocabulary_size):
                count = 0
                for doc_index in range(self.number_of_documents):
                    #self.topic_word_prob[topic_index][word_index] = self.term_doc_matrix[doc_index][word_index] * self.topic_prob[doc_index][topic_index][word_index]
                    count += self.term_doc_matrix[doc_index][word_index] * self.topic_prob[doc_index][topic_index][word_index]
                word_counts.append(count)
            #self.topic_word_prob = normalize(self.topic_word_prob)
            self.topic_word_prob[topic_index, :] = normalize(np.array(word_counts).reshape(1,-1))


    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################
        total = 0
        topic_total = 0
        for doc_index in range(self.number_of_documents):
            for word_index in range(self.vocabulary_size):
                topic_total = 0
                for topic_index in range(number_of_topics):
                    topic_total += self.document_topic_prob[doc_index][topic_index] * self.topic_word_prob[topic_index][word_index]
                total += self.term_doc_matrix[doc_index][word_index] * np.log(topic_total)
        self.likelihoods.append(total)

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = None

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            # ############################
            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)
            if current_likelihood and abs(current_likelihood - self.likelihoods[-1]) < epsilon:
                print("converged!!")
                break

            current_likelihood = self.likelihoods[-1]
        print("log likelihoods!!")
        print(self.likelihoods)



def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 160
    epsilon = 0.00001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
