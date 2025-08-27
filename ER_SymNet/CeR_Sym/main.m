clear; clc; close all;

%% 1. 数据加载
load FPHA_train_label    % 训练集标签
load FPHA_val_label      % 验证集标签  
load FPHA_train_seq      % 训练集序列
load FPHA_val_seq        % 验证集序列

Train_labels = train_labels;
Test_labels = val_labels;
Train_data = train_seq;
Test_data = val_seq;

%% 2. 模qww
% 型参数设置
params = setModelParameters();

%% 3. 训练阶段


% 计算协方差矩阵
cov_train = computeCov(Train_data);

% 训练网络并提取特征
[Train_features, T_1, W_1] = train(cov_train, Train_labels, params);

%% 4. 测试阶段

% 计算测试数据协方差矩阵
cov_test = computeCov(Test_data);

% 测试特征提取
Test_features = test(cov_test, T_1, W_1, params);

%% 5. 分类
[accuracy, ClassRate] = collaborativeDiscriminativeClassifier(Train_features, Test_features, ...
                                                             Train_labels, Test_labels, ...
                                                             params.deta, params.lamda2);

%% 6. 结果输出
fprintf('\n==================== 结果 ====================\n');
fprintf('正确分类的测试样本数: %d\n', accuracy);
fprintf('分类准确率: %.2f%%\n', ClassRate * 100);
fprintf('===============================================\n');