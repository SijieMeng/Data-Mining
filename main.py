import numpy as np
from sklearn.neural_network import MLPRegressor

np.set_printoptions(threshold=np.inf, suppress=True)
# 读取所有的TXT数据
buyer_basic_info = np.loadtxt('project/buyer_basic_info.txt')
buyer_historical_category15_money = np.loadtxt('project/buyer_historical_category15_money.txt')
buyer_historical_category15_quantity = np.loadtxt('project/buyer_historical_category15_quantity.txt')
key_product_IDs = np.loadtxt('project/key_product_IDs.txt')
product_distribution_training_set = np.loadtxt('project/product_distribution_training_set.txt')
product_features = np.loadtxt('project/product_features.txt')
trade_info_training = np.loadtxt('project/trade_info_training.txt')


# 整体销售数量预测,训练模型
# def AllQuantity(buyer_basic_info):
#     clf1 = MLPRegressor()
#     Quantity = buyer_basic_info[:, 1]
#     buyer_basic_info = np.delete(buyer_basic_info, 1, axis=1)
#     clf1.fit(buyer_basic_info, Quantity)
#     # 预测未来29的销量
#     Predit = (buyer_basic_info[buyer_basic_info.shape[0] - 1, 0] + 1) * np.ones(29) + np.arange(29)
#     Predit = np.reshape(Predit, [29, 1])
#     temp = buyer_basic_info[randint(low=0, high=buyer_basic_info.shape[1], size=29)]
#     temp = np.delete(temp, 0, axis=1)
#     PreditData = np.hstack((Predit, temp))
#     result = clf1.predict(PreditData)
#     result = np.hstack((Predit, np.reshape(result, [29, 1])))
#     print(result)
#     return result


def EveryProduct(product_distribution_training_set):
    # 创建训练的类
    clf2 = MLPRegressor()
    # 数据处理与变量命名
    ProductLabel = product_distribution_training_set[:, 0]
    product_distribution_training_set = np.delete(product_distribution_training_set, 0, 1)
    y = product_distribution_training_set.T
    X = np.arange(product_distribution_training_set.shape[1])
    X = np.reshape(X, [X.__len__(), 1])
    # 拟合训练
    clf2.fit(X, y)
    # 产生预测数据
    Predit = product_distribution_training_set.shape[1] + np.arange(29)
    Predit = np.reshape(Predit, [29, 1])
    # 产出预测结果
    result = clf2.predict(Predit)
    ProductLabel = np.reshape(ProductLabel, [ProductLabel.__len__(), 1])
    result1 = np.sum(result, axis=1)
    result1 = np.reshape(result1, [result1.__len__(), 1])
    result2 = np.hstack((ProductLabel, result.T))
    # 产生时间数据
    Year = np.arange(118, 118 + 29)
    Year = np.reshape(Year, [Year.__len__(), 1])
    result1 = np.hstack((Year, result1))
    # 输出结果
    print(result1, '\n')
    print(result2, '\n')
    return result1, result2


if __name__ == '__main__':
    # 调取函数
    result1, result2 = EveryProduct(product_distribution_training_set)
    # 保存为txt数据文件
    np.savetxt('result1.txt', result1, fmt=['%s'] * result1.shape[1])
    np.savetxt('result2.txt', result2, fmt=['%s'] * result2.shape[1])
