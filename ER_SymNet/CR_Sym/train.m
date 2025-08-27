function [Train_features, T_1, W_1] = train(cov_train, Train_labels, params)
%% 特征谱空间加权SymNet-v2网络训练函数
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

    T_1 = learnFirstLayerWeightsSpectral(cov_train, Train_labels, params);

    first_layer_output = firstLayerForwardSpectral(cov_train, T_1, params);


    W_1 = learnSecondLayerWeightsSpectral(first_layer_output, params);

    Train_features = fullForwardPassSpectral(cov_train, T_1, W_1, params);

end


function T_1 = learnFirstLayerWeightsSpectral(cov_train, Train_labels, params)
%% 学习第一层SPD映射权重 - 使用特征谱空间加权

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
    e_values = e_values(order, order);
    
    % 获取对角线值并开平方根
    diagonal_values = diag(e_values);
    diagonal_values = sqrt(diagonal_values);
    e_values = diagonal_values;
    
    % 特征谱空间加权
    R = rank(cov_all);
    e_vectors = applySpectralSpaceWeighting(e_vectors, e_values, R, params, 1);
    
    % 构建映射权重
    T_1 = cell(1, params.num_layers_1);
    for i = 1:params.num_layers_1
        start_idx = (i-1) * params.p_dim_1 + 1;
        end_idx = i * params.p_dim_1;
        T_1{i} = e_vectors(:, start_idx:end_idx);
    end
end


function first_layer_output = firstLayerForwardSpectral(cov_train, T_1, params)
%% 第一层前向传播 - 特征谱空间加权版本

    rectified_data_cell = cell(1, length(cov_train));
    all_rectified_maps = [];
    
    for i = 1:length(cov_train)
        single_sample_tr = cov_train{i};
        second_layer_singletr = cell(1, params.num_layers_1);
        
        % 各分支映射
        for j = 1:params.num_layers_1
            second_layer_singletr{j} = T_1{j}' * single_sample_tr * T_1{j};
        end
        
        % 第一层整流
        for k = 1:params.num_layers_1
            mid = second_layer_singletr{k};
            idx1 = mid <= 0;
            idx2 = mid > -params.eta_1;
            idx = idx1 & idx2;
            mid(idx) = -params.eta_1;
            [U, V, D] = svd(mid);
            [a, ~] = size(V);
            tol_1 = trace(V) * params.eps_1;
            for l = 1:a
                if V(l, l) <= tol_1
                    V(l, l) = tol_1;
                end
            end
            second_layer_singletr{k} = U * V * D';
        end
        
        rectified_data_cell{i} = second_layer_singletr;
        
        % 收集所有整流后的映射
        for s = 1:size(second_layer_singletr, 2)
            rectified_maps_matrix(:, :, s) = second_layer_singletr{s};
        end
        
        m = size(second_layer_singletr, 2);
        all_rectified_maps(:, :, (i-1)*m+1:i*m) = rectified_maps_matrix;
    end
    
    first_layer_output = struct();
    first_layer_output.rectified_data_cell = rectified_data_cell;
    first_layer_output.all_rectified_maps = all_rectified_maps;
end


function W_1 = learnSecondLayerWeightsSpectral(first_layer_output, params)
%% 学习第二层SPD映射权重 - 特征谱空间加权版本

    all_rectified_maps = first_layer_output.all_rectified_maps;
    
    % 计算第二层协方差
    maps_sum = zeros(params.p_dim_1, params.p_dim_1);
    for i = 1:size(all_rectified_maps, 3)
        maps_sum = maps_sum + all_rectified_maps(:, :, i);
    end
    mean_maps = maps_sum / size(all_rectified_maps, 3);
    
    Sum_CovMaps = zeros(params.p_dim_1, params.p_dim_1);
    for j = 1:size(all_rectified_maps, 3)
        diff = all_rectified_maps(:, :, j) - mean_maps;
        Sum_CovMaps = Sum_CovMaps + diff' * diff;
    end
    Sum_CovMaps = Sum_CovMaps / (size(all_rectified_maps, 3) - 1);
    
    % 特征分解
    [s_vectors, s_values] = eig(Sum_CovMaps);
    [~, order] = sort(diag(-s_values));
    s_vectors = s_vectors(:, order);
    s_values = s_values(order, order);
    
    % 获取对角线值并开平方根
    diagonal_values2 = diag(s_values);
    diagonal_values = sqrt(diagonal_values2);
    e_values = diagonal_values;
    
    % 特征谱空间加权
    R2 = rank(Sum_CovMaps);
    s_vectors = applySpectralSpaceWeighting(s_vectors, e_values, R2, params, 2);
    
    % 构建第二层权重
    W_1 = cell(1, params.num_layers_2);
    for k = 1:params.num_layers_2
        start_idx = (k-1) * params.p_dim_2 + 1;
        end_idx = k * params.p_dim_2;
        W_1{k} = s_vectors(:, start_idx:end_idx);
    end
end


function Train_features = fullForwardPassSpectral(cov_train, T_1, W_1, params)
%% 完整的训练前向传播 - 特征谱空间加权版本

    final_train = cell(1, length(cov_train));
    
    % 第一层处理
    first_layer_output = firstLayerForwardSpectral(cov_train, T_1, params);
    rectified_data_cell = first_layer_output.rectified_data_cell;
    
    for l = 1:length(cov_train)
        temp_ch = rectified_data_cell{l};
        final_train_branch = zeros(params.p_dim_2^2 * params.num_layers_2, params.num_layers_1);
        
        for r = 1:size(temp_ch, 2)
            temp1 = temp_ch{r};
            transfer_each = cell(1, params.num_layers_2);
            
            % 第二层映射
            for s = 1:params.num_layers_2
                transfer_each{s} = W_1{s}' * temp1 * W_1{s};
            end
            
            % 第二层整流
            for k = 1:params.num_layers_2
                mid = transfer_each{k};
                idx1 = mid <= 0;
                idx2 = mid > -params.eta_2;
                idx = idx1 & idx2;
                mid(idx) = -params.eta_2;
                [U, V, D] = svd(mid);
                [a, ~] = size(V);
                tol_2 = trace(V) * params.eps_2;
                for l1 = 1:a
                    if V(l1, l1) <= tol_2
                        V(l1, l1) = tol_2;
                    end
                end
                transfer_each{k} = U * V * D';
            end
            
            % 对数映射
            log_map = cell(1, params.num_layers_2);
            all_trace = zeros(1, params.num_layers_2);
            
            for p = 1:params.num_layers_2
                [u, v, w] = svd(transfer_each{p});
                logv = log(diag(v));
                log_map{p} = u * diag(logv) * w';
                
                [~, V, ~] = svd(log_map{p});
                all_trace(p) = trace(V);
            end
            
            % 计算权重并组合
            T_2 = all_trace / sum(all_trace);
            temp_final = [];
            
            for o = 1:params.num_layers_2
                temp2 = reshape(T_2(o) * log_map{o}, [], 1);
                temp_final = [temp_final; temp2];
            end
            
            final_train_branch(:, r) = temp_final;
        end
        
        final_train{l} = final_train_branch(:);
    end
    
    Train_features = final_train;
end


function weighted_vectors = applySpectralSpaceWeighting(vectors, eigenvalues, R, params, layer_num)
%% 应用特征谱空间加权
%
% 输入:
%   vectors - 特征向量
%   eigenvalues - 特征值
%   R - 矩阵秩
%   params - 参数结构体
%   layer_num - 层数(1或2)

    % 根据层数选择参数
    if layer_num == 1
        m = params.m;
        miu = params.miu;
        noise_cutoff = params.noise_cutoff_1;
        useless_start = params.useless_start_1;
    else
        miu = params.miu2;
        noise_cutoff = params.noise_cutoff_2;
        
        % 动态计算第二层的m值
        n = length(eigenvalues);
        if mod(n, 2) == 1
            middle_index = floor(n / 2) + 1;
        else
            middle_index = n / 2;
        end
        
        for i = 1:n
            constant = eigenvalues(middle_index) + miu * (eigenvalues(middle_index) - eigenvalues(R));
            if eigenvalues(i) < constant
                m = i;
                break;
            end
        end
    end
    
    % 计算特征谱空间权重参数
    if m > 1 && m <= length(eigenvalues) && eigenvalues(1) ~= eigenvalues(m)
        e_values_m = eigenvalues(m);
        e_values_1 = eigenvalues(1);
        
        aerfa = (e_values_1 * e_values_m * (m-1)) / (e_values_1 - e_values_m);
        bierta = (m * e_values_m - e_values_1) / (e_values_1 - e_values_m);
    else
        % 如果参数计算有问题，使用简单的权重
        aerfa = 1;
        bierta = 0;
    end
    
    weighted_vectors = vectors;
    
    % 主要空间加权
    for i = 1:min(m, length(eigenvalues))
        if eigenvalues(i) > 0
            weighted_vectors(:, i) = (1/sqrt(eigenvalues(i))) * vectors(:, i);
        end
    end
    
    % 噪声空间加权
    if layer_num == 1
        for i = m+1:min(noise_cutoff, length(eigenvalues))
            lada = aerfa / (i + bierta);
            if lada > 0
                weighted_vectors(:, i) = (1/sqrt(lada)) * vectors(:, i);
            end
        end
        
        % 无用空间加权
        for i = useless_start:min(R, length(eigenvalues))
            lada = aerfa / (R + bierta + 1);
            if lada > 0
                weighted_vectors(:, i) = (1/sqrt(lada)) * vectors(:, i);
            end
        end
    else
        for i = m+1:min(noise_cutoff, length(eigenvalues))
            lada = aerfa / (i + bierta);
            if lada > 0
                weighted_vectors(:, i) = (1/sqrt(lada)) * vectors(:, i);
            end
        end
    end
end