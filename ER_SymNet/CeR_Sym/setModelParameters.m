function params = setModelParameters()
%% 设置SymNet-v2模型的所有参数
%
% 输出:
%   params - 包含所有模型参数的结构体

    % 网络结构参数
    params.num_layers_1 = 3;        % 第一层分支数 m1
    params.p_dim_1 = 20;            % 第一层降维维度 d_m1
    params.num_layers_2 = 4;        % 第二层分支数 m2  
    params.p_dim_2 = 5;             % 第二层降维维度 d_m2

    % 整流层参数
    params.eps_1 = 4e-3;            % 第一层激活阈值 ε1
    params.eps_2 = 1e-3;            % 第二层激活阈值 ε2
    params.eta_1 = 1e-6;            % 第一层正定性保证参数 η1
    params.eta_2 = 1e-6;            % 第二层正定性保证参数 η2

    % 特征谱加权参数
    params.m = 5; %5                   % 主要空间维度
    params.z = 25;  %25                % 噪声空间截止维度
    params.miu = 1;                 % 权重计算参数

    % 分类器参数
    params.deta = 0.02;             % RBF核参数
    params.lamda2 = 0.2;            % 正则化参数
end