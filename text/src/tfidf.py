
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn
import sklearn.neural_network
from src.classifier import Classifier


class TFIDF(Classifier):

    def __init__(self, dataset, options = None):
        super().__init__(dataset, options)
        self.name = "tfidf"

    def _preprocess_options(self, options):
        """Preprocess options.

        Default values are override.
        """

        default = {
            "min_df": 0.01,
            "use_idf": True,
            "analyzer": "word",
            "stop_words": [],
            "ngram_range": (1,1),
            "max_features": 1000,
            "sublinear_tf": True
        }

        return {**default, **options}



    def train(self, tfidf_options =None, id =0 ):
        if tfidf_options == None:
            tfidf_options = self._preprocess_options({})
        self.clf = sklearn.neural_network.MLPClassifier(verbose=True,early_stopping=True, random_state = 44+id)
        self.vectorizer = TfidfVectorizer(**tfidf_options)
        (datas, labels) = self.dataset.get_train()
        data = self.vectorizer.fit_transform(datas)
        print("scaling")
        self.scaler = None
        #self.scaler = sklearn.preprocessing.StandardScaler()
        #data = self.scaler.fit_transform(data.toarray())
        print("MLP learning")
        self.clf.fit(data, labels)


    def predict(self, data):

        data = self.vectorizer.transform(data)
        if self.scaler is not None:
            data = self.scaler.transform(data.toarray())

        pred = self.clf.predict(data)
        proba =self.clf.predict_proba(data)

        rem = self.clf.out_activation_

        self.clf.out_activation_='identity'
        logits = self.clf._forward_pass_fast(data)
        self.clf.out_activation_= rem

        if self.scaler is None:
            data = data.toarray()

        return pred, proba, logits, data
