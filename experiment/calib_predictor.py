import pystruct
import numpy as np
from recal.structured_learners import mean_field,gibbs_sampling


class CalibPredictor(object):
    

    """docstring for CalibPredictor"""
    def __init__(self, X, Y, ssvm):
        super(CalibPredictor, self).__init__()
        self.X = X
        self.Y = Y
        self.ssvm = ssvm
        self.model = ssvm.model
        self.map_algs = ['bp','dd','trw','lf','alphaexp', 'ad3']

    # use mean field do marginal inference
    def marginal_inference(self, x):
        node_features, edges, edge_features = x
        num_nodes = node_features.shape[0]
        w = self.ssvm.w
        up = np.exp(self.model._get_unary_potentials(x, w))
        pp = np.exp(self.model._get_pairwise_potentials(x, w))
        return mean_field(up, pp, edges, max_iter=100)

    # use different map inference algorithms
    def map_inference(self, x, alg='ad3'):
        assert alg in self.map_algs
        if alg == 'ad3':
            return pystruct.inference.inference_ad3(
            self.ssvm.model._get_unary_potentials(x, self.ssvm.w),
            self.ssvm.model._get_pairwise_potentials(x, self.ssvm.w),
            x[1]
            )
        else:
            return pystruct.inference.inference_ogm(
            self.ssvm.model._get_unary_potentials(x, self.ssvm.w),
            self.ssvm.model._get_pairwise_potentials(x, self.ssvm.w),
            x[1],
            alg=alg
            )

    # get all the marginals
    def all_marginal(self):
        return [self.marginal_inference(x) for x in X]

    def gen_Y_best(self):
        margins = self.all_marginal()
        self.Y_best = [np.argmax(m,axis=1) for m in margins]

    def labelize(self):
        # if not self.Y_best:
        #     self.Y_map = [map_inference(x) for x in self.X]
        Y = self.Y
        Y_best = self.Y_best
        self.Z = np.array([1 if y_i == y_best_i else 0 
                            for Y_i ,Y_best_i in zip(Y,Y_best) 
                            for y_i, y_best_i in zip(Y_i ,Y_best_i)])

    def 

    








