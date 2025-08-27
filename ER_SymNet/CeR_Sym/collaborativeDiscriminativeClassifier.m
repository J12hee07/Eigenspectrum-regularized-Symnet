function [accuracy, ClassRate] = collaborativeDiscriminativeClassifier(Train_features, Test_features, Train_labels, Test_labels, deta, lamda2)
%% 协同判别分类器
%
% 输入:
%   Train_features - 训练特征
%   Test_features - 测试特征
%   Train_labels - 训练标签
%   Test_labels - 测试标签
%   deta - RBF核参数
%   lamda2 - 正则化参数
%
% 输出:
%   accuracy - 正确分类的样本数
%   ClassRate - 分类准确率

    % 构建训练核矩阵
    K_xx = computeKernelMatrix(Train_features, Train_features, deta);
    
    % SVD分解
    [U, V] = svd(K_xx);
    u = sqrt(V);
    phi = (U * u)';
    inv_phi = pinv(phi');
    
    accuracy = 0;
    unique_labels = unique(Test_labels);

    % 对每个测试样本进行分类
    for i = 1:length(Test_labels)
        if mod(i, 100) == 0
        end
        
        K_xy = computeKernelMatrix(Train_features, Test_features(i), deta);
        phila = inv_phi * K_xy;
        
        tmp = phi' * phi;
        omega = pinv(tmp + lamda2 * eye(size(tmp))) * phi' * phila;
        
        % 计算各类重构误差
        errors = zeros(1, length(unique_labels));
        for j = 1:length(unique_labels)
            class_indices = find(unique_labels(j) == Train_labels);
            phi_c = phi(:, class_indices);
            W_c = omega(class_indices, 1);
            errors(j) = norm(phila - phi_c * W_c, 2)^2 / sum(W_c .* W_c);
        end
        
        % 选择最小误差对应的类别
        [~, min_idx] = min(errors);
        predicted_label = unique_labels(min_idx);
        
        if predicted_label == Test_labels(i)
            accuracy = accuracy + 1;
        end
    end
    
    ClassRate = accuracy / length(Test_labels);
end
function K = computeKernelMatrix(features1, features2, deta)
%% 计算RBF核矩阵
%
% 输入:
%   features1 - 第一组特征
%   features2 - 第二组特征
%   deta - 核参数
%
% 输出:
%   K - 核矩阵

    if iscell(features2)
        K = zeros(length(features1), length(features2));
        for i = 1:length(features1)
            for j = 1:length(features2)
                dist = norm(features1{i} - features2{j});
                K(i, j) = exp(-deta * dist^2);
            end
        end
    else
        K = zeros(length(features1), 1);
        for i = 1:length(features1)
            dist = norm(features1{i} - features2);
            K(i, 1) = exp(-deta * dist^2);
        end
    end
end