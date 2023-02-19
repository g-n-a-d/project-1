from utils.tf_idf import TF_IDF

def APISeq_split_TFIDF(df, test_size=0.12):
    words = list(map(lambda s: s.split(','), list(df['api'])))
    label = list(df['class'])
    tf_idf = TF_IDF(words)
    x_train, x_test, y_train, y_test = train_test_split(words, label, test_size=test_size)
    return x_train, x_test, y_train, y_test, tf_idf
