import torch
import torch.nn as nn
from collections import namedtuple
# from helper import ALE_MM_loss_onehot_y, SYNC_LOSS_onehot_y, LSEP_Loss

def max_margin_loss_onehot_y(scores, onehot_y, margin=1):
    N, C = onehot_y.size()
    score_y = (scores * onehot_y).sum(1)
    score_y = score_y.view(N, 1).expand(N, C)

    margin = (1-onehot_y)*margin
    loss = scores + margin - score_y
    loss = torch.clamp(loss, min=0)

    return loss

ALE_N = 100
ALE_coef = torch.zeros(ALE_N)
base = 0.0
for i in range(1, ALE_N):
    base += 1.0/i
    ALE_coef[i] = base/i

def ALE_MM_loss_onehot_y(scores, onehot_y, margin=1):
    from torch.autograd import Variable
    global ALE_coef

    mml = max_margin_loss_onehot_y(scores, onehot_y, margin=margin)

    num_positive = (mml>0).sum(1).long().data
    if scores.is_cuda:
        ALE_coef = ALE_coef.cuda()
    weight = ALE_coef.gather(0, num_positive)
    weight = Variable(weight).view(-1, 1)

    loss = mml*weight
    return loss

def SYNC_LOSS_onehot_y(scores, onehot_y, distance):
    N, C = onehot_y.size()
    score_y = (scores * onehot_y).sum(1)
    score_y = score_y.view(N, 1).expand(N, C)

    margin = torch.mm(onehot_y, distance)
    loss = scores + margin - score_y

    loss, _ = loss.max(1)
    loss = torch.clamp(loss, min=0)

    return loss

def LSEP_Loss_one_hot_y(scores, onehot_y):
    N, C = onehot_y.size()
    
    score_y = (scores * onehot_y).sum(1)
    score_y = score_y.view(N, 1).expand(N, C)

    loss = torch.exp(scores-score_y) + 1
    loss = torch.log(loss)
    return loss

class MLPHat(nn.Module):

    Result = namedtuple('MLPHat_Results', "feature score loss label")

    def __init__(self, feature_size, emb_size, 
                loss_type='ALE',
                ):

        super(MLPHat, self).__init__()

        self.feature_size = feature_size
        self.emb_size = emb_size
        self.loss_type = loss_type
        self.p_positive = 0.5
        print 'Using %s Loss' % self.loss_type

        self.input_batchnorm=nn.BatchNorm1d(feature_size, affine=True)        
        self.linear = nn.Linear(feature_size, emb_size)
        self.feature_batchnorm=nn.BatchNorm1d(emb_size, affine=True)        

    def forward(self, input, emb_yT):
        """
        input: image feature, NxE 
        emb_yT: transposed class embeddings, ExC
        """
        input = self.input_batchnorm(input)
        feature = self.linear(input)
        feature = self.feature_batchnorm(feature)
        score = torch.mm(feature, emb_yT)
        return feature, score

    def compute_result(self, inputs, y, emb_yT=None, emb_distance=None, sync_simi=None):
        """
        emb_distance, emb_simi: not used, just to compact with SYNC
        """
        emb_x, scores = self.forward(inputs, emb_yT)

        if self.loss_type == "ALE":
            loss = ALE_MM_loss_onehot_y(scores, y)
        elif self.loss_type == "F0T":
            loss = LSEP_Loss_one_hot_y(scores, y)

        return MLPHat.Result(emb_x, scores, loss, y)


class SYNCHat(nn.Module):

    Result = namedtuple('SYNCHat_Results', "feature score loss label")

    def __init__(self, feature_size, emb_size,
                num_V,
                ):

        super(MLPHat, self).__init__()

        self.feature_size = feature_size
        self.emb_size = emb_size

        self.input_batchnorm=nn.BatchNorm1d(feature_size, affine=True)        
        self.linear = nn.Linear(feature_size, emb_size)
        self.feature_batchnorm = nn.BatchNorm1d(emb_size, affine=True)

        V = torch.FloatTensor(num_V, emb_size) # VxE
        self.V = nn.Parameter(V)

    def forward(self, input, simi):
        """
        input: image feature, NxE 
        simi: similarity weight between the Phony classes V and the true classes, CxV
        """
        input = self.input_batchnorm(input)
        feature = self.net(input)
        feature = self.feature_batchnorm(feature)

        W = torch.mm(simi, self.V) 
        score = torch.mm(feature, torch.t(W))

        return feature, score

    def compute_result(self, inputs, y, emb_yT=None, emb_distance=None, sync_simi=None):
        """
        emb_yT: not used, just to compact with SYNC
        emb_distance: precomputed distance between each embeddings
        """
        emb_x, scores = self.forward(inputs, sync_simi)
        loss = SYNC_LOSS_onehot_y(scores, y, emb_distance)

        return MLPHat.Result(emb_x, scores, loss, y)


