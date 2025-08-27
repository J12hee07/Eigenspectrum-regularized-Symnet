function [Train_features, T_1, W_1] = train(cov_train, Train_labels, params)
%% SymNet-v2网络训练函数
%
% 输入:
%   cov_train - 训练数据的协方差矩阵
%   Train_labels - 训练标签
%   params - 模型参数
%
% 输出:
%   Train_features - 训练特征
%   T_1 - 第一层映射权重
%   W_1 - 第二层映射权重

  
    T_1 = learnFirstLayerWeights(cov_train, Train_labels, params);

    first_layer_output = firstLayerForward(cov_train, T_1, params);

   
    W_1 = learnSecondLayerWeights(first_layer_output, params);

   
    Train_features = fullForwardPass(cov_train, T_1, W_1, params);
    
end


function T_1 = learnFirstLayerWeights(cov_train, Train_labels, params)
%% 学习第一层SPD映射权重

    % 计算训练数据的总体协方差
    cov_sum = zeros(63, 63);
    for k = 1:length(Train_labels)
        cov_sum = cov_sum + cov_train{k};
    end
    mean_train = cov_sum / length(Train_labels);
    
    % 计算散布矩阵
    cov_all = zeros(63, 63);
    for i = 1:length(Train_labels)
        diff = cov_train{i} - mean_train;
        cov_all = cov_all + diff' * diff;
    end
    cov_all = cov_all / (length(Train_labels) - 1);
    
    % 特征分解
    [e_vectors, e_values] = eig(cov_all);
    [~, order] = sort(diag(-e_values));
    e_vectors = e_vectors(:, order);
    e_values = sqrt(diag(e_values(order, order)));
    
    % 特征谱空间加权
    R = rank(cov_all);
    e_vectors = applySpectralWeighting(e_vectors, e_values, params.m, params.z, R);
    
    % 构建映射权重
    T_1 = cell(1, params.num_layers_1);
    for i = 1:params.num_layers_1
        start_idx = (i-1) * params.p_dim_1 + 1;
        end_idx = i * params.p_dim_1;
        T_1{i} = e_vectors(:, start_idx:end_idx);
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


function W_1 = learnSecondLayerWeights(first_layer_data, params)
%% 学习第二层SPD映射权重

    % 收集所有第一层输出
    all_matrices = [];
    for i = 1:length(first_layer_data)
        for j = 1:length(first_layer_data{i})
            if isempty(all_matrices)
                all_matrices = first_layer_data{i}{j};
            else
                all_matrices = cat(3, all_matrices, first_layer_data{i}{j});
            end
        end
    end
    
    % 计算第二层协方差
    maps_sum = zeros(size(all_matrices, 1), size(all_matrices, 2));
    for i = 1:size(all_matrices, 3)
        maps_sum = maps_sum + all_matrices(:, :, i);
    end
    mean_maps = maps_sum / size(all_matrices, 3);
    
    Sum_CovMaps = zeros(size(mean_maps));
    for i = 1:size(all_matrices, 3)
        diff = all_matrices(:, :, i) - mean_maps;
        Sum_CovMaps = Sum_CovMaps + diff' * diff;
    end
    Sum_CovMaps = Sum_CovMaps / (size(all_matrices, 3) - 1);
    
    % 特征分解和加权
    [s_vectors, s_values] = eig(Sum_CovMaps);
    [~, order] = sort(diag(-s_values));
    s_vectors = s_vectors(:, order);
    s_values = sqrt(diag(s_values(order, order)));
    
    % 应用加权（简化版）
    for i = 1:params.m
        if i <= length(s_values) && s_values(i) > 0
            s_vectors(:, i) = s_vectors(:, i) / sqrt(s_values(i));
        end
    end
    
    % 构建第二层权重
    W_1 = cell(1, params.num_layers_2);
    for k = 1:params.num_layers_2
        start_idx = (k-1) * params.p_dim_2 + 1;
        end_idx = k * params.p_dim_2;
        W_1{k} = s_vectors(:, start_idx:end_idx);
    end
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


function weighted_vectors = applySpectralWeighting(e_vectors, e_values, m, z, R)
%% 特征谱空间加权函数

    weighted_vectors = e_vectors;
    
    % 防止除零错误
    if e_values(1) == e_values(m)
        % 如果特征值相等，使用简单的归一化
        for i = 1:min(m, length(e_values))
            if e_values(i) > 0
                weighted_vectors(:, i) = e_vectors(:, i) / sqrt(e_values(i));
            end
        end
        return;
    end
    
    % 计算权重参数
    e_values_m = e_values(m);
    e_values_1 = e_values(1);
    aerfa = (e_values_1 * e_values_m * (m-1)) / (e_values_1 - e_values_m);
    bierta = (m * e_values_m - e_values_1) / (e_values_1 - e_values_m);
    
    % 主要空间加权
    for i = 1:m
        if e_values(i) > 0
            weight = 1 / sqrt(e_values(i));
            weighted_vectors(:, i) = weight * e_vectors(:, i);
        end
    end
    
    % 噪声空间加权
    for i = m+1:min(z, length(e_values))
        lada = aerfa / (i + bierta);
        if lada > 0
            weight = 1 / sqrt(lada);
            weighted_vectors(:, i) = weight * e_vectors(:, i);
        end
    end
    
    % 无用空间加权
    for i = z+1:min(R, length(e_values))
        lada = aerfa / (R + bierta + 1);
        if lada > 0
            weight = 1 / sqrt(lada);
            weighted_vectors(:, i) = weight * e_vectors(:, i);
        end
    end
end