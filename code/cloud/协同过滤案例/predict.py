import var
import numpy as np

def predict_baseline(user, item):
    '''baseline'''
    prediction = var.item_mean[item] + var.user_mean[user] - var.all_mean
    return prediction

def predict_itemCF(user, item):
    '''item-item协同过滤算法'''
    nzero = var.ratings[user].nonzero()[0]
    prediction = var.ratings[user, nzero].dot(var.item_similarity[item, nzero])\
                / sum(var.item_similarity[item, nzero])
    return prediction

def predict_userCF(user, item):
    '''user-user协同过滤算法'''
    nzero = var.ratings[:,item].nonzero()[0]
    prediction = (var.ratings[nzero, item]).dot(var.user_similarity[user, nzero])\
                / sum(var.user_similarity[user, nzero])
    if np.isnan(prediction):
        baseline = var.user_mean + var.item_mean[item] - var.all_mean
        prediction = baseline[user]
    return prediction

def predict_itemCF_baseline(user, item):
    '''结合baseline的item-item CF算法'''
    nzero = var.ratings[user].nonzero()[0]
    baseline = var.item_mean + var.user_mean[user] - var.all_mean
    prediction = (var.ratings[user, nzero] - baseline[nzero]).dot(var.item_similarity[item, nzero])\
                / sum(var.item_similarity[item, nzero]) + baseline[item]
    return prediction 

def predict_userCF_baseline(user, item):
    '''结合baseline的user-user协同过滤算法,预测rating'''
    nzero = var.ratings[:,item].nonzero()[0]
    baseline = var.user_mean + var.item_mean[item] - var.all_mean
    prediction = (var.ratings[nzero, item] - baseline[nzero]).dot(var.user_similarity[user, nzero])\
                / sum(var.user_similarity[user, nzero]) + baseline[user]
    if np.isnan(prediction): prediction = baseline[user]
    return prediction

def predict_itemCF_bias(user, item):
    '''结合baseline的item-item CF算法,预测rating'''
    nzero = var.ratings[user].nonzero()[0]
    baseline = var.item_mean + var.user_mean[user] - var.all_mean
    prediction = (var.ratings[user, nzero] - baseline[nzero]).dot(var.item_similarity[item, nzero])\
                / sum(var.item_similarity[item, nzero]) + baseline[item]
    if prediction > 5: prediction = 5
    if prediction < 1: prediciton = 1
    return prediction

def predict_topkCF_item(user, item, k=20):
    '''top-k CF算法,以item-item协同过滤为基础，结合baseline,预测rating'''
    nzero = var.ratings[user].nonzero()[0]
    baseline = var.item_mean + var.user_mean[user] - var.all_mean
    choice = nzero[var.item_similarity[item, nzero].argsort()[::-1][:k]]
    prediction = (var.ratings[user, choice] - baseline[choice]).dot(var.item_similarity[item, choice])\
                / sum(var.item_similarity[item, choice]) + baseline[item]
    if prediction > 5: prediction = 5
    if prediction < 1: prediction = 1
    return prediction

def predict_topkCF_user(user, item, k=30):
    '''top-k CF算法,以user-user协同过滤为基础，结合baseline,预测rating'''    
    nzero = var.ratings[:,item].nonzero()[0]
    choice = nzero[var.user_similarity[user, nzero].argsort()[::-1][:k]]
    baseline = var.user_mean + var.item_mean[item] - var.all_mean
    prediction = (var.ratings[choice, item] - baseline[choice]).dot(var.user_similarity[user, choice])\
                / sum(var.user_similarity[user, choice]) + baseline[user]
    if np.isnan(prediction): prediction = baseline[user]
    if prediction > 5: prediction = 5
    if prediction < 1: prediction = 1
    return prediction

def predict_normCF_item(user, item, k=20):
    '''在topK的基础上对item采用归一化的相似度矩阵'''
    nzero = var.ratings[user].nonzero()[0]
    baseline = var.item_mean + var.user_mean[user] - var.all_mean
    choice = nzero[var.item_similarity_norm[item, nzero].argsort()[::-1][:k]]
    prediction = (var.ratings[user, choice] - baseline[choice]).dot(var.item_similarity_norm[item, choice])\
                / sum(var.item_similarity_norm[item, choice]) + baseline[item]
    if prediction > 5: prediction = 5
    if prediction < 1: prediction = 1
    return prediction

def predict_normCF_user(user, item, k=30):
    '''在topK的基础上对user采用归一化的相似度矩阵'''
    nzero = var.ratings[:,item].nonzero()[0]
    choice = nzero[var.user_similarity_norm[user, nzero].argsort()[::-1][:k]]
    baseline = var.user_mean + var.item_mean[item] - var.all_mean
    prediction = (var.ratings[choice, item] - baseline[choice]).dot(var.user_similarity_norm[user, choice])\
                / sum(var.user_similarity_norm[user, choice]) + baseline[user]
    if np.isnan(prediction): prediction = baseline[user]
    if prediction > 5: prediction = 5
    if prediction < 1: prediction = 1
    return prediction

def predict_blend(user, item, k1=20, k2=30, alpha=0.6):
    '''融合模型'''
    prediction1 = predict_topkCF_item(user, item, k1)
    prediction2 = predict_topkCF_user(user, item, k2)
    prediction = alpha * prediction1 + (1-alpha) * prediction2
    if prediction > 5: prediction = 5
    if prediction < 1: prediction = 1
    return prediction