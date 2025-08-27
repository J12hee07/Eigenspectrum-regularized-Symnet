function [L_train, y_test] = convertFeaturesToMatrix(Train_features, Test_features)
%% 将特征从cell格式转换为矩阵格式
%
% 输入:
%   Train_features - 训练特征(cell格式)
%   Test_features - 测试特征(cell格式)
%
% 输出:
%   L_train - 训练特征矩阵
%   y_test - 测试特征矩阵


    % 训练集转换为矩阵
    L_train = [];
    for i = 1:length(Train_features)
        L_train = [L_train, Train_features{i}];
    end
    

    % 测试集转换为矩阵
    y_test = [];
    for i = 1:length(Test_features)
        y_test = [y_test, Test_features{i}];
    end
    
end