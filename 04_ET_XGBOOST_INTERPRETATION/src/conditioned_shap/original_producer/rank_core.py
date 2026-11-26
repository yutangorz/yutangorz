"""Aligned local-response scores, not a new attribution estimator."""
import numpy as np
from scipy.stats import rankdata


def lime_contrast(delta_x, training_sd, coefficient):
    x,s,b=map(lambda a:np.asarray(a,float),(delta_x,training_sd,coefficient))
    if x.ndim!=2 or x.shape[1]!=9 or s.shape!=(9,) or b.shape!=(9,):
        raise ValueError('Expected n x9 contrasts and two9-vectors')
    if not all(np.isfinite(a).all() for a in [x,s,b]) or (s<=0).any():
        raise ValueError('Finite inputs and positive scales required')
    return x/s*b


def shap_contrast(perturbed, parent):
    v,b=np.asarray(perturbed,float),np.asarray(parent,float)
    if v.ndim!=2 or v.shape[1]!=9 or b.shape!=(9,) or not np.isfinite(v).all() or not np.isfinite(b).all():
        raise ValueError('Malformed SHAP contrast')
    return v-b


def recover_legacy_coefficient_blocks(frame, features):
    """run019 appended ET then XGB without a model column; preserve full blocks.

    Never average/drop duplicate features. The caller must replay each block
    against the source model-labelled predictions before adopting the ET block.
    """
    p=len(features)
    if len(frame) not in [p,2*p]:raise ValueError('Unexpected legacy coefficient block count')
    blocks=[]
    for start in range(0,len(frame),p):
        block=frame.iloc[start:start+p].set_index('feature')
        if not block.index.is_unique or set(block.index)!=set(features):raise ValueError('Malformed legacy block')
        v=block.loc[features,'coefficient_per_training_SD'].to_numpy(float)
        if not np.isfinite(v).all():raise ValueError('Nonfinite legacy coefficient')
        blocks.append(v)
    return blocks


def paired_bootstrap(scores, seed=20261030, repeats=2000):
    """scores[parent,method,feature]; resample the same parents for all methods."""
    scores=np.asarray(scores,float)
    if scores.ndim!=3 or not len(scores) or not np.isfinite(scores).all() or (scores<0).any():
        raise ValueError('Nonnegative finite parent importance scores required')
    rng=np.random.default_rng(seed)
    ii=rng.integers(0,len(scores),size=(repeats,len(scores)))
    return scores[ii].mean(axis=1)


def row_spearman(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or a.ndim!=2:raise ValueError('Matching 2D matrices required')
    a=rankdata(a,axis=1);b=rankdata(b,axis=1);a-=a.mean(axis=1,keepdims=True);b-=b.mean(axis=1,keepdims=True)
    den=np.sqrt((a*a).sum(axis=1)*(b*b).sum(axis=1))
    return np.divide((a*b).sum(axis=1),den,out=np.full(len(a),np.nan),where=den>0)
