function params = setSpectralWeightingParameters()
%% 设置特征谱空间加权SymNet-v2模型的所有参数
%
% 输出:
%   params - 包含所有模型参数的结构体

    % 网络结构参数
    params.num_layers_1 = 3;        % 第一层分支数
    params.p_dim_1 = 20;            % 第一层降维维度
    params.num_layers_2 = 4;        % 第二层分支数
    params.p_dim_2 = 5;             % 第二层降维维度

    % 整流层参数
    params.eps_1 = 4e-3;            % 第一层激活阈值
    params.eps_2 = 1e-3;            % 第二层激活阈值
    params.eta_1 = 1e-6;            % 第一层正定性保证参数
    params.eta_2 = 1e-6;            % 第二层正定性保证参数

    % 特征谱空间加权参数
    params.m = 5;                   % 主要空间维度（第一层）
    params.miu = 1;                 % 阈值计算参数（第一层）
    params.miu2 = 1;                % 阈值计算参数（第二层）
    params.noise_cutoff_1 = 45;     % 第一层噪声空间截止点
    params.useless_start_1 = 46;    % 第一层无用空间起始点
    params.noise_cutoff_2 = 20;     % 第二层噪声空间截止点

    % 分类器参数
    params.lamda1 = 10;             % 欧式CRC正则化参数

end