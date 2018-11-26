import numpy as np

def softmax(x, t=1.0, axis=1):
    x = x/t
    x = x - x.max(axis, keepdims=True)
    x = np.exp(x)
    x = x / x.sum(axis, keepdims=1)
    return x

def split(M, L):
    ll = M[:L, :L]
    lu = M[:L, L:]
    ul = M[L:, :L]
    uu = M[L:, L:]
    return ll, lu, ul, uu

class LPBASE(object):

    def get_transform(self):
        return self.puu_inv.dot(self.pul)

    @staticmethod
    def process_p(p, n, S):
        ll, lu, ul, uu = split(p, S)
        uu = np.identity(uu.shape[0]) - (1-n)*uu
        uu_inv = np.linalg.inv(uu)
        return ul, uu_inv

    def _predict_prob(self, seen, hu):
      score = (1-self.n)*(self.pul.dot(seen)) + self.n*hu
      score = self.puu_inv.dot(score)
      return score          

class LPincr(LPBASE):
    dot_product = lambda x, y: x.dot(y.T)

    def __init__(self, seen_feat, label, t=1.0, simi_func=None):
        self.t = t

        # num_seen = base_feat.shape[0]
        self.num_seen = seen_feat.shape[0]
        self.seen_feat = seen_feat
        self.label = label

        self.simi_func = simi_func if simi_func else lambda x, y: x.dot(y.T)

    def predict(self, unseen_feat, n=0, hu=None, nothide_idx=None):
        featAll = np.concatenate([self.seen_feat, unseen_feat], 0)
        simi = self.simi_func(featAll, featAll)
        p = softmax(simi, t=self.t, axis=1)
        S = self.seen_feat.shape[0]
        pul, puu_inv = self.process_p(p, n, S)

        if not nothide_idx:
            raw_score = pul.dot(self.label)
        else:
            p = pul[:, nothide_idx]
            l = self.label[nothide_idx]
            raw_score = p.dot(l)

        if n > 0:
            raw_score = (1-n)*(raw_score) + n*hu

        score = puu_inv.dot(raw_score)

        return score

    def update_by_unseen(self, unseen_feat, n=0, hu=None, nothide_idx=None):

        score = self.predict(unseen_feat, n=n, hu=hu, nothide_idx=nothide_idx)

        self.seen_feat = np.concatenate([self.seen_feat, unseen_feat])
        self.label = np.concatenate([self.label, score], 0)
        self.num_seen = self.seen_feat.shape[0]

        return score

    def incre_predict(self, new_feat):
        affn = self.simi_func(new_feat, self.seen_feat)
        self_dot = np.diag(self.simi_func(new_feat, new_feat)).reshape(-1, 1)
        affn = np.concatenate([affn, self_dot], axis=1)
        p_new = softmax(affn, self.t)

        score = p_new[:, :-1].dot(self.label)
        div = 1-p_new[:,-1] 
        if len(score.shape) > 1:
            div = div.reshape(-1, 1)

        score = score / div
        return score

def beta_normalize(fit_score, real_score, subset, outlier_filter=None):
    from scipy.stats import beta
    beta_score = np.zeros_like(real_score)
    for i in subset:
        data = fit_score[:, i]
        if outlier_filter:
            data = outlier_filter(data)
        offset = (data.max() - data.min())/10
        a, b, loc, scale = beta.fit(data, floc=data.min()-offset, fscale=data.max()-data.min()+2*offset)
        real_data = real_score[:, i]
        cdf = beta.cdf(real_data, a, b, loc, scale)
        beta_score[:, i] = cdf
    return beta_score


def approximate_score(feature, label, fix_feature, fix_label, t=1, num_fold=5):
    # print "approximate fold", num_fold
    from sklearn.model_selection import KFold
    kfold = KFold(num_fold, shuffle=True)
    scores = []

    for train_id, test_id in kfold.split(feature):
        sX = feature[train_id]
        uX = feature[test_id]
        sL = label[train_id]

        sX = np.concatenate([fix_feature, sX])
        sL = np.concatenate([fix_label, sL])
        lp_ = LPincr(sX, sL, t=t)

        s = lp_.predict(uX)

        scores.append(s)

    scores = np.concatenate(scores, 0)
    return scores