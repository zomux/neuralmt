import numpy
from collections import Counter
from theano import tensor as T
from deepy import NeuralLayer, EPSILON


def bleu_stats(hypothesis, reference):
    yield len(hypothesis)
    yield len(reference)
    for n in xrange(1, 5):
        s_ngrams = Counter([tuple(hypothesis[i:i + n]) for i in xrange(len(hypothesis) + 1 - n)])
        r_ngrams = Counter([tuple(reference[i:i + n]) for i in xrange(len(reference) + 1 - n)])
        yield sum((s_ngrams & r_ngrams).values())
        yield max(len(hypothesis) + 1 - n, 0)

def bleu(stats):
    stats = numpy.atleast_2d(numpy.asarray(stats))[:, :10].sum(axis=0)
    if not all(stats):
        return 0
    c, r = stats[:2]
    if c == 0: return 0.
    log_bleu_prec = sum([numpy.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]) / 4.
    return numpy.exp(min(0, 1 - float(r) / c) + log_bleu_prec) * 100

def smoothed_bleu(stats):
    c, r = stats[:2]
    if c == 0: return 0.
    log_bleu_prec = sum([numpy.log((1 + float(x)) / (1 + y)) for x, y in zip(stats[2::2], stats[3::2])]) / 4.
    return numpy.exp(min(0, 1 - float(r) / c) + log_bleu_prec) * 100


class ResetLayer(NeuralLayer):

    def __init__(self, var, dim=0):
        super(ResetLayer, self).__init__("reset")
        self.var = var
        self.output_dim = dim

    def compute_tensor(self, x):
        return self.var


class LogProbLayer(NeuralLayer):

    def compute_tensor(self, x):
        prob = T.clip(x, EPSILON, 1.0 - EPSILON)
        log_prob = - T.log(prob)
        return log_prob