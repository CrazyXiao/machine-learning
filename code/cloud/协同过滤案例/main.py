import pandas as pd
import numpy as np
import var
import predict as pre
import utils

print('初始化变量...')
names = ['user_id', 'item_id', 'rating', 'timestamp']
direct = 'dataset/ml-100k/'
trainingset_files = (direct + name for name in ('u1.base', 'u2.base', 'u3.base', 'u4.base', 'u5.base'))
testset_files = (direct + name for name in ('u1.test', 'u2.test', 'u3.test', 'u4.test', 'u5.test'))

if __name__ == '__main__':

    rmse_baseline = []
    rmse_itemCF = []
    rmse_userCF = []
    rmse_itemCF_baseline = []
    rmse_userCF_baseline = []
    rmse_itemCF_bias = []
    rmse_topkCF_item = []
    rmse_topkCF_user = []
    rmse_normCF_item = []
    rmse_normCF_user = []
    rmse_blend = []
    i = 0
    nums = 5
    for trainingset_file, testset_file in zip(trainingset_files, testset_files):
        i += 1
        print('------ 第%d/%d组样本 ------' % (i, nums))
        df = pd.read_csv(trainingset_file, sep='\t', names=names)
        
        var.ratings = np.zeros((var.n_users, var.n_items))
        print('载入训练集' + trainingset_file)
        for row in df.itertuples():
            var.ratings[row[1]-1, row[2]-1] = row[3]
        
        print('训练集规模为 %d' % len(df))

        sparsity = utils.cal_sparsity()
        print('训练集矩阵密度为 {:4.2f}%'.format(sparsity))
        
        print('计算训练集各项统计数据...')
        utils.cal_mean()

        print('计算相似度矩阵...')
        var.user_similarity = utils.cal_similarity(kind='user')
        var.item_similarity = utils.cal_similarity(kind='item')
        var.user_similarity_norm = utils.cal_similarity_norm(kind='user')
        var.item_similarity_norm = utils.cal_similarity_norm(kind='item')
        print('计算完成')
        
        print('载入测试集' + testset_file)
        test_df = pd.read_csv(testset_file, sep='\t', names=names)
        predictions_baseline = []
        predictions_itemCF = []
        predictions_userCF = []
        predictions_itemCF_baseline = []
        predictions_userCF_baseline = []
        predictions_itemCF_bias = []
        predictions_topkCF_item = []
        predictions_topkCF_user = []
        predictions_normCF_item = []
        predictions_normCF_user = []
        predictions_blend = []
        targets = []
        print('测试集规模为 %d' % len(test_df))
        print('测试中...')
        for row in test_df.itertuples():
            user, item, actual = row[1]-1, row[2]-1, row[3]
            predictions_baseline.append(pre.predict_baseline(user, item))
            predictions_itemCF.append(pre.predict_itemCF(user, item))
            predictions_userCF.append(pre.predict_userCF(user, item))
            predictions_itemCF_baseline.append(pre.predict_itemCF_baseline(user, item))
            predictions_userCF_baseline.append(pre.predict_userCF_baseline(user, item))
            predictions_itemCF_bias.append(pre.predict_itemCF_bias(user, item))
            predictions_topkCF_item.append(pre.predict_topkCF_item(user, item, 20))
            predictions_topkCF_user.append(pre.predict_topkCF_user(user, item, 30))
            predictions_normCF_item.append(pre.predict_normCF_item(user, item, 20))
            predictions_normCF_user.append(pre.predict_normCF_user(user, item, 30))
            predictions_blend.append(pre.predict_blend(user, item, 20, 30, 0.7))
            targets.append(actual)
    
        rmse_baseline.append(utils.rmse(np.array(predictions_baseline), np.array(targets)))
        rmse_itemCF.append(utils.rmse(np.array(predictions_itemCF), np.array(targets)))
        rmse_userCF.append(utils.rmse(np.array(predictions_userCF), np.array(targets)))
        rmse_itemCF_baseline.append(utils.rmse(np.array(predictions_itemCF_baseline), np.array(targets)))
        rmse_userCF_baseline.append(utils.rmse(np.array(predictions_userCF_baseline), np.array(targets)))
        rmse_itemCF_bias.append(utils.rmse(np.array(predictions_itemCF_bias), np.array(targets)))
        rmse_topkCF_item.append(utils.rmse(np.array(predictions_topkCF_item), np.array(targets)))
        rmse_topkCF_user.append(utils.rmse(np.array(predictions_topkCF_user), np.array(targets)))
        rmse_normCF_item.append(utils.rmse(np.array(predictions_normCF_item), np.array(targets)))
        rmse_normCF_user.append(utils.rmse(np.array(predictions_normCF_user), np.array(targets)))
        rmse_blend.append(utils.rmse(np.array(predictions_blend), np.array(targets)))
        print('测试完成')
    print('------ 测试结果 ------')
    print('各方法在交叉验证下的RMSE值:')
    print('baseline:           %.4f' % np.mean(rmse_baseline))
    print('itemCF:             %.4f' % np.mean(rmse_itemCF))
    print('userCF:             %.4f' % np.mean(rmse_userCF))
    print('itemCF_baseline:    %.4f' % np.mean(rmse_itemCF_baseline))
    print('userCF_baseline:    %.4f' % np.mean(rmse_userCF_baseline)) 
    print('itemCF_bias:        %.4f' % np.mean(rmse_itemCF_bias))
    print('topkCF(item, k=20): %.4f' % np.mean(rmse_topkCF_item))
    print('topkCF(user, k=30): %.4f' % np.mean(rmse_topkCF_user))
    print('normCF(item, k=20): %.4f' % np.mean(rmse_normCF_item))
    print('normCF(user, k=30): %.4f' % np.mean(rmse_normCF_user))
    print('blend (alpha=0.7):  %.4f' % np.mean(rmse_blend))
    print('交叉验证运行完成')
    