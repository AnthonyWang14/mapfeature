import numpy as np
import multiprocessing as mp
import pystruct
from sumproduct import Variable, Factor, FactorGraph

# ----------------------------------------------------------------------------

class CRFPredictor(object):
	def __init__(self, ssvm, model):
		self.ssvm = ssvm
		self.model = model

	def predict_proba(self, X, Y):
		return np.array([self._proba(x,y) for x,y in zip(X,Y)])

	def predict_marginals(self, X):
		return [self._marginals(x) for x in X]

	def pseudomarginals(self, X, Y_map=None):
		if not Y_map:
			Y_map = self.predict_map(X)

		return [self._pseudomarginals(x, y_map) for x,y_map in zip(X, Y_map)]

	def fit(self, X, Y):
		self.ssvm.fit(X,Y)
		return self

	def unnormalized_probability(self, X, Y):
		return [self._one_unnnormalized_probability(x,y) for x,y in zip(X,Y)]

	def score(self, X, Y):
		Y_pred = self.predict(X)
		model = self.ssvm.model
		loss = sum([model.loss(y_pred,y) for (y_pred, y) in zip(Y_pred,Y)])
		tot = sum(len(y) for y in Y)

		return (tot-loss)/tot
	
	def predict(self, X):
		return [self._predict(x) for x in X]

	def predict_via_marginal_decoding(self, X):
		return [self._marginal_decoding(x) for x in X]

	def _one_unnnormalized_probability(self, x, y):
		return np.exp(np.dot(self.ssvm.w, self.ssvm.model.joint_feature(x,y)))

	def _marginal_decoding(self, x):
		P = self._marginals(x)

		# marginal decoding:
		y_map = np.argmax(P, axis=1)

		return y_map

	def _marginal_2nd_best_decoding(self, x):
		P = self._marginals(x)

		# marginal decoding:
		y_map = np.argmax(P, axis=1)

		# determine second best:
		P_max = np.max(P, axis=1) # score of most likely label in MAP assignemnt
		i_closest = np.argmin(P_max) # index of MAP label that's least confident
		P_i_sorted = np.argsort(P[i_closest,:]) # sorted labels in that dimension
		y2 = P_i_sorted[-2] # second most probable label in that dimension

		y_map2 = y_map.copy()
		y_map2[i_closest] = y2

		return y_map2

	def _predict_2nd_best(self,X):
		return [self._marginal_2nd_best_decoding(x) for x in X]

	### subclasses need to implement:

	def _marginals(self, x):
		raise Exception("Method not implemented")

	def _proba(self, x):
		raise Exception("Method not implemented")

	def _predict(self, x):
		raise Exception("Method not implemented")


class ChainCRFPredictor(CRFPredictor):
	"""Wraps a chain CRF, adds belief propagation and marginal decoding"""
	
	def __init__(self, chain_ssvm, model):
		# super(ChainCRFPredictor, self).__init__(chain_ssvm, model)
		self.ssvm = chain_ssvm
		self.model = model

	def predict(self, X):
		return [self._marginal_decoding(x) for x in X]

	def predict_map(self, X):
		return self.ssvm.predict(X)

	def predict_2nd_best(self,X):
		return [self._marginal_2nd_best_decoding(x) for x in X]

	def margins(self, X):
		Y1 = self.predict(X)
		Y2 = self._predict_2nd_best(X)
		S1 = self.unnormalized_probability(X, Y1)
		S2 = self.unnormalized_probability(X, Y2)

		return np.array([s1 - s2 for (s1,s2) in zip(S1, S2)])

	def count_marginals(self,X):
		return np.sum([len(x) for x in X])

	def _marginals0(self, x):
		w = self.ssvm.w
		U = np.exp(self.model._get_unary_potentials(x, w))
		P = np.exp(self.model._get_pairwise_potentials(x, w))

		assert not isinstance(P,tuple)

		# create a factor graph for this instance
		g = FactorGraph(silent=True)
		v = list()
		uf = list()
		pf = list()

		# create vertices and unary potentials
		for i, x_i in enumerate(x):
			name = 'v%d' % i
			v.append(Variable(name, self.y_dim))
			g.add(v[i])

			table = np.exp(w_x.dot(x_i).reshape(self.y_dim,1))
			factor_name = 'f%d' % i
			uf.append(Factor(factor_name, table))

			g.add(uf[i])
			g.append(factor_name, v[i])

		# create edges and pairwise potentials
		for i in xrange(len(x)-1):
			name = 'f%d%d' % (i, i+1)
			pf.append(Factor(name, np.exp(w_y)))
			
			g.add(pf[i])
			g.append(name, v[i])
			g.append(name, v[i+1])

		# run belief propagation
		try:
			g.compute_marginals()

			# compute probabilities
			# P = np.vstack((w_x.dot(x_i).reshape(self.y_dim) for x_i in x))
			P = np.vstack((g.nodes['v%d'%i].marginal() for i in xrange(len(x))))
		except TypeError:
			P = np.random.rand(len(x), self.y_dim)

		return P

	def _proba(self, x, y):
		w = self.ssvm.w
		U = np.exp(self.model._get_unary_potentials(x, w))
		P = np.exp(self.model._get_pairwise_potentials(x, w))

		assert not isinstance(P,tuple)

		unnorm_prob = np.exp(self.ssvm.w.dot(self.ssvm.model.joint_feature(x,y)))
		_, Z = forwards_backwards(U,P.T)

		return unnorm_prob / Z

	def _marginals(self, x):
		w = self.ssvm.w
		U = np.exp(self.model._get_unary_potentials(x, w))
		P = np.exp(self.model._get_pairwise_potentials(x, w))

		assert not isinstance(P,tuple)

		M, Z = forwards_backwards(U,P.T)

		return M

	def _pseudomarginals(self, x, y_map):
		""" WARNING: Not tested; transposes may be wrong """
		n_nodes = len(y_map)
		w = self.ssvm.w
		U = np.exp(self.model._get_unary_potentials(x, w))
		P = np.exp(self.model._get_pairwise_potentials(x, w))

		Q = np.empty(U.shape)

		for i in xrange(n_nodes):
			t = U[i].copy()
			if i < n_nodes-1:
				t += (P.T)[:,y_map[i+1]]
			if i > 0:
				t += P[:,y_map[i-1]]
			t = np.exp(t)
			Q[i] = t / np.sum(t)

		return Q

	def _predict(self, x):
		return self.ssvm.predict([x])

class GraphCRFPredictor(CRFPredictor):
	"""Wraps a graph CRF, adds LBP, marginal decoding, mean-field, etc."""

	def __init__(self, graph_ssvm, model):
		# super(GraphCRFPredictor, self).__init__(graph_ssvm, model)
		self.ssvm = graph_ssvm
		self.model = model

	def predict_via_mf(self, X):
		return [np.argmax(self._marginals_via_mf(x),axis=1) for x in X]

	def predict_via_bp(self, X):
		return [np.argmax(self._marginals_via_bp(x),axis=1) for x in X]

	def predict_via_gibbs(self, X):
		out = list()
		for i, x in enumerate(X):
			print i
			out.append(np.argmax(self._marginals_via_gibbs(x),axis=1))
		return out
		# return [np.argmax(self._marginals_via_gibbs(x),axis=1) for x in X]

	def predict_map(self, X, method='ad3'):
		if method=='ad3':
			return [self._predict_via_dual_decomp(x) for x in X]
		else:
			return [self._predict_via_ogm(x,method) for x in X]

	def pseudomarginals(self, X, Y_map=None):
		if not Y_map:
			Y_map = self.predict_map(X)

		return [self._pseudomarginals(x, y_map) for x,y_map in zip(X, Y_map)]

	# for one sample x, y_map
	def _pseudomarginals(self, x, y_map):
		w = self.ssvm.w
		up = np.exp(self.model._get_unary_potentials(x, w))
		pp = np.exp(self.model._get_pairwise_potentials(x, w))

		edges = x[1]

		return pseudomarginals(up, pp, edges, y_map)

	def _predict_via_dual_decomp(self, x):
		return pystruct.inference.inference_ad3(
			self.model._get_unary_potentials(x, self.ssvm.w),
			self.model._get_pairwise_potentials(x, self.ssvm.w),
			x[1]
		)

	def _predict_via_ogm(self, x, method):
		return pystruct.inference.inference_ogm(
			self.model._get_unary_potentials(x, self.ssvm.w),
			self.model._get_pairwise_potentials(x, self.ssvm.w),
			x[1],
			alg=method
		)
	def _predict(self, x):
		return self._marginal_decoding(x)

	def _marginals(self, x, method='mf'):
		if method == 'mf':
			# print 'use mf'
			return self._marginals_via_mf(x)
		elif method == 'bp':
			return self._marginals_via_bp(x)
		else:
			raise ValueError("Invalid method for producing marginals")

	def count_marginals(self,X):
		return np.sum([len(x[0]) for x in X])

	def _marginals_via_mf(self, x):
		node_features, edges, edge_features = x
		num_nodes = node_features.shape[0]

		w = self.ssvm.w

		up = np.exp(self.model._get_unary_potentials(x, w))
		pp = np.exp(self.model._get_pairwise_potentials(x, w))

		return mean_field(up, pp, edges, max_iter=100)

	def _marginals_via_bp(self, x):
		node_features, edges, edge_features = x
		num_nodes = node_features.shape[0]

		w = self.ssvm.w

		up = np.exp(self.model._get_unary_potentials(x, w))
		pp = np.exp(self.model._get_pairwise_potentials(x, w))

		return loopy_async_bp(up, pp, edges, max_iter=20)

	def _marginals_via_gibbs(self, x, Y0=None):
		node_features, edges, edge_features = x
		num_nodes = node_features.shape[0]

		w = self.ssvm.w

		up = np.exp(self.model._get_unary_potentials(x, w))
		pp = np.exp(self.model._get_pairwise_potentials(x, w))

		if Y0 == None:
			Y0 = self._predict_via_dual_decomp(x)

		return gibbs_sampling(up, pp, edges, Y0=Y0)


# ----------------------------------------------------------------------------
# Belief propagation

def _forwards(U, P):
	"""
	Forwards pass of forwards-backwards with unary potentials U and 
	pairwise potentials P.

	WARNING: There is no normalization 
			 and this will not work for long chains
	"""
	T, d = U.shape
	assert P.shape == (d,d)

	M = np.zeros((T,d)) # will store the messages from t-1 to t

	for t in xrange(T):
		if t == 0:
			M[t,:] = np.ones(d,)
		else:
			local_belief = M[t-1,:] * U[t-1,:]
			M[t,:] = np.dot(P, local_belief)
 
	return M

def _backwards(U, P):
	"""
	Backwards pass of forwards-backwards with unary potentials U and 
	pairwise potentials P.

	WARNING: There is no normalization 
			 and this will not work for long chains
	"""
	T, d = U.shape
	assert P.shape == (d,d)

	M = np.zeros((T,d)) # will store the messages from t+1 to t

	for t in reversed(xrange(0,T)):
		if t == T-1:
			M[t,:] = np.ones(d,)
		else:
			local_belief = M[t+1,:] * U[t+1,:]
			M[t,:] = np.dot(P.T, local_belief)

	return M

def forwards_backwards(U, P):
	"""
	Computes marginals using forwards-backwards.
	"""
	T, d = U.shape
	assert P.shape == (d,d)

	M_fwd = _forwards(U, P)
	M_bwd = _backwards(U, P)

	Z_fwd = M_fwd[T-1,:] * U[T-1]
	Z_bwd = M_bwd[0,:] * U[0]

	assert np.abs(np.sum(Z_fwd) - np.sum(Z_bwd)) / np.sum(Z_bwd) < 1e-6

	unnorm_marginals = (M_fwd[t,:]*M_bwd[t,:]*U[t,:] for t in xrange(T))
	norm_marginals = np.vstack((m / np.sum(m) for m in unnorm_marginals))

	return norm_marginals, np.sum(Z_fwd)

def loopy_async_bp(U, P, edges, max_iter = 1000):
	iter = 0
	n_nodes, n_states = U.shape

	# set initial belief to one
	beliefs = np.ones(U.shape)
	
	# initialize messages; messages[t][s] holds message s -> t
	messages = { t : dict() for t in xrange(n_nodes) }
	for v1, v2 in edges:
		messages[v1][v2] = np.ones(n_states,)
		messages[v2][v1] = np.ones(n_states,)

	ordered_edges = [(i, e, 'fwd') for i, e in enumerate(edges)] \
				  + [(i, e, 'bwd') for i, e in enumerate(reversed(edges))]

	max_err = 1
	while iter < max_iter:
		max_err = 0
		if iter % 100 == 0 and iter > 0 == 0: print iter

		for i, (v1, v2), direction in ordered_edges:
			if direction == 'fwd':
				edge_potential = P[i]
				s, t = v1, v2
			elif direction == 'bwd':
				edge_potential = P[i].T
				s, t = v2, v1
			else:
				raise ValueError("Invalid direction")
			
			# compute s -> t message
			belief = U[s].copy()
			for m in messages[s].values():
				belief *= m
			belief /= np.sum(belief)
			beliefs[s] = belief

			B_msg = belief / messages[s][t]
			new_message = np.dot(edge_potential, B_msg)
			max_err = np.max([max_err, np.linalg.norm(new_message - messages[t][s])])
			messages[t][s] = new_message

		# print max_err
		iter += 1

	for s, belief in enumerate(beliefs):
		# compute s -> t message
			belief = U[s].copy()
			for m in messages[s].values():
				belief *= m
			belief /= np.sum(belief)
			beliefs[s] = belief 

	return beliefs

def loopy_sync_bp(U, P, edges, 	max_iter=1000):
	""" Synchronous BP """
	iter = 0
	n_nodes, n_states = U.shape

	# set initial belief to one
	beliefs = { v : np.ones(n_states,) for v in xrange(n_nodes) }
	
	# initialize messages; messages[t][s] holds message s -> t
	messages = { t : dict() for t in xrange(n_nodes) }
	for v1, v2 in edges:
		messages[v1][v2] = np.ones(n_states,)
		messages[v2][v1] = np.ones(n_states,)

	while iter <= max_iter:
		if iter % 100 == 0: print iter

		# compute every message
		new_messages = { t : dict() for t in xrange(n_nodes) }
		for i, (v1, v2) in enumerate(edges):
			edge_potential = P[i].T
			
			# compute v1 -> v2 message
			B_msg = np.ones(n_states,)
			for v_in, m in messages[v1].iteritems():
				if v_in != v2: B_msg *= m
			msg = np.dot(edge_potential.T, B_msg * U[v1])
			msg /= np.sum(msg)
			new_messages[v2][v1] = msg
			# print '...', new_messages[v2][v1]

			# compute v2 -> v1 message
			B_msg = np.ones(n_states,)
			for v_in, m in messages[v2].iteritems():
				if v_in != v1: B_msg *= m
			msg = np.dot(edge_potential, B_msg * U[v2])
			msg /= np.sum(msg)
			new_messages[v1][v2] = msg

		# update node beliefs:
		for t in xrange(n_nodes):
			belief = U[t].copy()
			for m in new_messages[t].values():
				belief *= m
			belief /= np.sum(belief)
			beliefs[t] = belief

		# update messages:
		messages = new_messages

		iter += 1

	return beliefs

def mean_field(U, P, edges, max_iter=100):
	iter = 0
	n_nodes, n_states = U.shape

	edge_potentials = { v : dict() for v in xrange(n_nodes) }
	
	for i, (v1, v2) in enumerate(edges):
		edge_potentials[v1][v2] = P[i]
		edge_potentials[v2][v1] = P[i].T

	Q = U.copy()

	max_err = 1
	while iter < max_iter and max_err > 1e-4:
		max_err = 0
		for i in xrange(n_nodes):
			t = U[i].copy()
			for n, E in edge_potentials[i].iteritems():
				t += E.dot(Q[n])
				# print t
			t = np.exp(t)
			Q_new = t / np.sum(t)
			max_err = np.max([max_err, np.linalg.norm(Q_new - Q[i])])
			Q[i] = Q_new
		iter += 1

	return Q

def gibbs_sampling(U, P, edges, n_samples=15000, n_burn=10000, Y0=None):
	iter = 0
	n_nodes, n_states = U.shape

	edge_potentials = { v : dict() for v in xrange(n_nodes) }
	
	for i, (v1, v2) in enumerate(edges):
		edge_potentials[v1][v2] = P[i]
		edge_potentials[v2][v1] = P[i].T

	if Y0 is not None:
		Y_curr = Y0.copy()
	else:
		Y_curr = np.random.randint(n_states, size=(n_nodes,))

	samples = np.empty((n_nodes, n_states, n_samples))

	while iter < n_samples + n_burn:
		for i in xrange(n_nodes):
			t = U[i].copy()
			for n, E in edge_potentials[i].iteritems():
				t += E[:,Y_curr[n]]
			t = np.exp(t)
			t /= np.sum(t)
			choices = range(len(U[i]))
			j = _choice(choices, t)
			Y_curr[i] = j
		
		if iter >= n_burn:
			Y_curr_mat = np.zeros((n_nodes, n_states))
			Y_curr_mat[np.arange(n_nodes), Y_curr] = 1
			samples[:,:,iter-n_burn] = Y_curr_mat

		iter += 1

	Q = np.mean(samples, axis=2)
	return Q

def pseudomarginals(U,P,edges,Y_map):
	n_nodes, n_states = U.shape
	# print U.shape
	edge_potentials = { v : dict() for v in xrange(n_nodes) }
	
	for i, (v1, v2) in enumerate(edges):
		# P[i]shape: n_states*n_states
		edge_potentials[v1][v2] = P[i]
		edge_potentials[v2][v1] = P[i].T

	Q = np.empty(U.shape)

	for i in xrange(n_nodes):
		t = U[i].copy()
		for n, E in edge_potentials[i].iteritems():
			# print 'Y_map[n]', Y_map[n]
			t += E[:,Y_map[n]]
		t = np.exp(t)
		Q[i] = t / np.sum(t)
	return Q

def _choice(objects, weights):
	cs = np.cumsum(weights)
	idx = sum(cs < np.random.rand())
	return objects[idx]