from flask_restplus import Namespace, Resource, fields
from core.Word2VecTester import Word2VecTester

word2vec_tester_instance = None

api = Namespace('Word2VecTester', description='Trained Word2Vec testing related operations')

most_similar_words_payload = api.model('MostSimilarWordPayload', {
    'word': fields.String(required=True, description='Word to find similarity from.')
})

most_similar_words_response = api.model('MostSimilarWordsResponse', {
    'similar_words': fields.String(required=True, description='List of words most similar to given word')
})

similarity_between_words_payload = api.model('TwoWordSimilarityPayload', {
    'word1': fields.String(required=True, description='First Word'),
    'word2': fields.String(required=True, description='Second Word')
})

similarity_between_words_response = api.model('TwoWordSimilarityResponse', {
    'similarity': fields.Float(required=True, description='Similarity between the given two words.')
})

@api.route('/initialize')
class InitializeLatestModel(Resource):
    @api.doc('Initialize Latest trained model')
    def get(self):
        """Initialize Latest trained model."""
        global word2vec_tester_instance
        word2vec_tester_instance = Word2VecTester()
        return 'Success', 200


@api.route('/isInitialized')
class IsInitialized(Resource):
    """Checks if the model is initialized and ready to classify."""
    # @api.doc('basic mathematical computations')
    # @api.expect(classification_payload)
    # @api.marshal_with(classification_response, code=200)
    def get(self):
        """Checks if the model is initialized and ready to classify."""
        global word2vec_tester_instance
        if word2vec_tester_instance is None:
            return 'Not Initialized'
        else:
            return 'Initialized', 200


@api.route('/get_similar_words')
class TestWord2Vec(Resource):
    @api.doc('find similar words to given word')
    @api.expect(most_similar_words_payload)
    @api.marshal_with(most_similar_words_response)
    def post(self):
        """Finds most similar words to the given word."""
        global word2vec_tester_instance
        if word2vec_tester_instance is None:
            return 'Model uninitialized. Please initialize.'
        else:
            return {'similar_words': word2vec_tester_instance.most_similar_words(self.api.payload['word'])}


@api.route('/get_similarity_between_two_words')
class TestWord2Vec(Resource):
    @api.doc('find_similarity between two words')
    @api.expect(similarity_between_words_payload)
    @api.marshal_with(similarity_between_words_response)
    def post(self):
        """Finds most similar words to the given word."""
        global word2vec_tester_instance
        if word2vec_tester_instance is None:
            return 'Model uninitialized. Please initialize.'
        else:
            return {'similarity': word2vec_tester_instance.get_similarity_between_two_words(
                self.api.payload['word1'], self.api.payload['word2'])}