# lightgbm训练个贷违约训练集
训练集为48万条，特征值有3300个的个贷违约数据集，违约样本：非违约样本=2:100，用随机森林方法找出相关性最高的600个特征值，用lightgbm训练该训练集。
多次测试后，目前得到的效果最好的参数为：        

lgb = LGBMClassifier( 

        objective='binary', # 二分类问题
        metric='auc', # 评估指标为AUC
        n_estimators=3000, # 减小树的数量
        max_depth=5, # 减小树的最大深度
        num_leaves=31, # 减小叶子节点数
        learning_rate=0.03, # 减小学习率
        colsample_bytree=0.7, # 减小每棵树使用的特征比例
        subsample=0.8, # 减小每次迭代使用的样本比例
        reg_alpha=0.2, # 增大L1正则化系数
        reg_lambda=0.3, # 增大L2正则化系数
        min_child_samples=20, # 增大最小子样本数
        min_data_in_bin=6, # 增大每个节点最小数据量
        drop_rate=0.1, # 添加Dropout
        random_state=42, # 随机种子
        n_jobs=-1, # 使用全部CPU核心
        scale_pos_weight=scale_pos_weight
    )
