function Test_features = test(cov_test, T_1, W_1, params)
%% 特征谱空间加权SymNet-v2网络测试函数
%
% 输入:
%   cov_test - 测试数据的协方差矩阵
%   T_1 - 第一层映射权重
%   W_1 - 第二层映射权重
%   params - 模型参数
%
% 输出:
%   Test_features - 测试特征
    
    final_test = cell(1, length(cov_test));
    rectified_data_cell = cell(1, length(cov_test));
    
    % 第一层处理
    for i = 1:length(cov_test)
        single_sample_te = cov_test{i};
        second_layer_singlete = cell(1, params.num_layers_1);
        
        % 各分支映射
        for j = 1:params.num_layers_1
            second_layer_singlete{j} = T_1{j}' * single_sample_te * T_1{j};
        end
        
        % 第一层整流
        for k = 1:params.num_layers_1
            mid = second_layer_singlete{k};
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
            second_layer_singlete{k} = U * V * D';
        end
        
        rectified_data_cell{i} = second_layer_singlete;
    end
    
    % 第二层处理
    for l = 1:length(cov_test)
        temp_ch = rectified_data_cell{l};
        final_test_branch = zeros(params.p_dim_2^2 * params.num_layers_2, params.num_layers_1);
        
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
            
            final_test_branch(:, r) = temp_final;
        end
        
        final_test{l} = final_test_branch(:);
    end
    
    Test_features = final_test;

end