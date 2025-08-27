function Test_features = test(cov_test, T_1, W_1, params)
%% SymNet-v2网络测试函数
%
% 输入:
%   cov_test - 测试数据的协方差矩阵
%   T_1 - 第一层映射权重
%   W_1 - 第二层映射权重
%   params - 模型参数
%
% 输出:
%   Test_features - 测试特征

    Test_features = fullForwardPass(cov_test, T_1, W_1, params);
    
end


function features = fullForwardPass(cov_data, T_1, W_1, params)
%% 完整的前向传播函数

    features = cell(1, length(cov_data));
    
    for i = 1:length(cov_data)
        % 第一层处理
        first_layer_out = firstLayerForward(cov_data(i), T_1, params);
        
        % 第二层处理
        final_branch = [];
        for j = 1:params.num_layers_1
            branch_input = first_layer_out{1}{j};
            branch_output = secondLayerForward(branch_input, W_1, params);
            final_branch = [final_branch; branch_output(:)];
        end
        
        features{i} = final_branch;
    end
end


function rectified_data = firstLayerForward(cov_data, T_1, params)
%% 第一层前向传播

    rectified_data = cell(1, length(cov_data));
    
    for i = 1:length(cov_data)
        single_sample = cov_data{i};
        layer_outputs = cell(1, params.num_layers_1);
        
        % 各分支处理
        for j = 1:params.num_layers_1
            % 映射
            mapped = T_1{j}' * single_sample * T_1{j};
            
            % 整流
            rectified = rectifyMatrix(mapped, params.eps_1, params.eta_1);
            layer_outputs{j} = rectified;
        end
        
        rectified_data{i} = layer_outputs;
    end
end


function output = secondLayerForward(input_matrix, W_1, params)
%% 第二层前向传播

    transfer_each = cell(1, params.num_layers_2);
    
    % 各分支映射和整流
    for s = 1:params.num_layers_2
        mapped = W_1{s}' * input_matrix * W_1{s};
        transfer_each{s} = rectifyMatrix(mapped, params.eps_2, params.eta_2);
    end
    
    % 对数映射
    log_maps = cell(1, params.num_layers_2);
    all_trace = zeros(1, params.num_layers_2);
    
    for p = 1:params.num_layers_2
        [u, v, w] = svd(transfer_each{p});
        logv = log(diag(v));
        log_maps{p} = u * diag(logv) * w';
        
        [~, V, ~] = svd(log_maps{p});
        all_trace(p) = trace(V);
    end
    
    % 计算权重并组合
    T_2 = all_trace / sum(all_trace);
    output = [];
    
    for o = 1:params.num_layers_2
        weighted_map = T_2(o) * log_maps{o};
        vectorized = reshape(weighted_map, [], 1);
        output = [output; vectorized];
    end
end


function rectified = rectifyMatrix(matrix, eps, eta)
%% 矩阵整流函数

    % 处理负值
    idx1 = matrix <= 0;
    idx2 = matrix > -eta;
    idx = idx1 & idx2;
    matrix(idx) = -eta;
    
    % SVD分解和阈值处理
    [U, V, D] = svd(matrix);
    tol = trace(V) * eps;
    
    for i = 1:size(V, 1)
        if V(i, i) <= tol
            V(i, i) = tol;
        end
    end
    
    rectified = U * V * D';
end