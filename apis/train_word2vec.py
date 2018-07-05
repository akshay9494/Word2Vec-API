from flask_restplus import Namespace, Resource, fields
from core.Word2VecTrainer import Word2VecTrainer


api = Namespace('Word2VecTrainer', description='Word2Vec Training related operations')

word2vec_training_statistics = api.model('Word2VecTrainingStatistice', {
    'training_loss': fields.String(required=True, description='Loss after training'),
    'vocab_length': fields.String(required=True, description='Length'),
})


@api.route('/train_from_scratch')
class TrainWord2Vec(Resource):
    @api.doc('train-word2vec')
    @api.marshal_list_with(word2vec_training_statistics)
    def get(self):
        """Trains Word2Vec from scratch"""
        trainer = Word2VecTrainer()
        results = trainer.train_from_scratch()
        return results


@api.route('/train_on_existing')
class TrainWord2Vec(Resource):
    @api.doc('train-word2vec')
    @api.marshal_list_with(word2vec_training_statistics)
    def get(self):
        """Trains Word2Vec from scratch"""
        trainer = Word2VecTrainer()
        results = trainer.train_on_existing()
        return results
