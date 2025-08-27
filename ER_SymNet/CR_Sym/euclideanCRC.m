function [accuracy, ClassRate] = euclideanCRC(L_train, y_test, Train_labels, Test_labels, lamda1)
%% 欧式协同表示分类器
%
% 输入:
%   L_train - 训练特征矩阵
%   y_test - 测试特征矩阵
%   Train_labels - 训练标签
%   Test_labels - 测试标签
%   lamda1 - 正则化参数
%
% 输出:
%   accuracy - 正确分类的样本数
%   ClassRate - 分类准确率


    
    % 计算全局协同表示系数
    tmp = L_train' * L_train;
    omega = pinv(tmp + lamda1 * eye(size(tmp))) * L_train' * y_test;
    
    accuracy = 0;
    unique_labels = unique(Test_labels);

    
    for i = 1:length(Test_labels)
        if mod(i, 100) == 0

        end
        
        Y = y_test(:, i);
        ERR1 = zeros(2, length(unique_labels));
        
        % 计算各类重构误差
        for j = 1:length(unique_labels)
            class_indices = find(unique_labels(j) == Train_labels);
            phi_c = L_train(:, class_indices);
            W_c = omega(class_indices, i);
            
            % 计算重构误差
            ERR = norm(Y - phi_c * W_c, 2)^2 / sum(W_c .* W_c);
            
            ERR1(1, j) = ERR;
            ERR1(2, j) = unique_labels(j);
        end
        
        % 选择最小误差对应的类别
        [~, index] = sort(ERR1(1, :));
        if ERR1(2, index(1)) == Test_labels(i)
            accuracy = accuracy + 1;
        end
    end
    
    ClassRate = accuracy / length(Test_labels);

end