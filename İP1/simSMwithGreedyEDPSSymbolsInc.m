%% NxM MIMO Setup Movable Antenna Position Optimization for MIMO IM
clear;
clc;


%%Simulation
plotBERvsSNR()

%==========================================================================
% Simulate transmission
%==========================================================================
function [ber_best, berMod_best, berAntenna_best, ber_worst, berMod_worst, berAntenna_worst ...
    , numErrors_best, numBits] = simulateSpatialModulationRayleighML(numTx, numRx, modOrder, num_iters, snrDb, channel_config)
    % Function to simulate Spatial Modulation for MIMO systems with Rayleigh fading
    % using Joint Maximum Likelihood Detection

    % Input Parameters:
    % numTx - Number of transmit antennas
    % numRx - Number of receive antennas
    % modOrder - Modulation order (e.g., 4 for QPSK)
    % num_iters - Number of iterations for transmission
    
    Lt = channel_config.Lt;
    Lr = channel_config.Lr;
    lambda = channel_config.lambda;
    kappa = channel_config.kappa;
    rowIndices_t = channel_config.rowIndices_t;
    rowIndices_r = channel_config.rowIndices_r;
    t_tilde_grid = channel_config.t_tilde_grid;
    r_tilde_grid = channel_config.r_tilde_grid;
    HcolIndices = channel_config.HcolIndices;
    % Initializations
    numBitsPerSymbol = log2(modOrder) + log2(numTx);
    bitsAll = [];
    bitsAntennaAll = [];
    bitsModAll = [];
    rxBitsAll_best = [];
    rxBitsAll_worst = [];
    rxBitsAntennaAll_best = [];
    rxBitsAntennaAll_worst = [];
    rxBitsModAll_best = [];
    rxBitsModAll_worst = [];

    % Possible symbols and their corresponding bit labels
    allSymbols = qammod(0:modOrder-1, modOrder, 'UnitAveragePower', true);
    allBitsMod = de2bi(0:modOrder-1, log2(modOrder), 'left-msb');

    for iter = 1:num_iters
        % Generate random bits for one symbol
        bits = randi([0 1], numBitsPerSymbol, 1);

        % Reshape bits into two parts
        bitsTx = bits';
        bitsAntenna = bitsTx(1:log2(numTx));
        bitsMod = bitsTx(log2(numTx) + 1:end);

        % Convert bits to antenna indices and modulation symbols
        antennaIndex = bi2de(bitsAntenna, 'left-msb') + 1;
        symbol = qammod(bi2de(bitsMod, 'left-msb'), modOrder, 'UnitAveragePower', true);

        % Transmit symbol
        txSignal = zeros(1, numTx);
        txSignal(antennaIndex) = symbol;

        
        % Get channels
        [H_best, H_rand, H_worst] = get_selected_channels(numTx, numRx, Lt, Lr, lambda, kappa, rowIndices_t, rowIndices_r, t_tilde_grid, r_tilde_grid, HcolIndices, allSymbols);

        rxSignal_best = H_best * txSignal.';
        rxSignal_worst = H_worst * txSignal.';

        % Add AWGN noise
        noisePower = 10^(-snrDb/10);
        noise = sqrt(noisePower/2) * (randn(numRx, 1) + 1j*randn(numRx, 1));
        
        rxSignal_best = rxSignal_best + noise;
        rxSignal_worst = rxSignal_worst + noise;

        [rxBits_best, rxBitsMod_best, rxBitsAntenna_best] = ML_detect(rxSignal_best, H_best, numTx, modOrder, allSymbols, allBitsMod);
        [rxBits_worst, rxBitsMod_worst, rxBitsAntenna_worst] = ML_detect(rxSignal_worst, H_worst, numTx, modOrder, allSymbols, allBitsMod);

        % Record bits
        bitsAll = [bitsAll; bits'];
        bitsAntennaAll = [bitsAntennaAll; bitsAntenna];
        bitsModAll = [bitsModAll; bitsMod];
        
        rxBitsAll_best = [rxBitsAll_best; rxBits_best];
        rxBitsModAll_best = [rxBitsModAll_best; rxBitsMod_best];
        rxBitsAntennaAll_best = [rxBitsAntennaAll_best; rxBitsAntenna_best];

        rxBitsAll_worst = [rxBitsAll_worst; rxBits_worst];
        rxBitsModAll_worst = [rxBitsModAll_worst; rxBitsMod_worst];
        rxBitsAntennaAll_worst = [rxBitsAntennaAll_worst; rxBitsAntenna_worst];
    end

    % Calculate errors and BER
    numErrors_best = sum(bitsAll(:) ~= rxBitsAll_best(:));
    numErrorsMod_best = sum(bitsModAll(:) ~= rxBitsModAll_best(:));
    numErrorsAntenna_best = sum(bitsAntennaAll(:) ~= rxBitsAntennaAll_best(:));

    numErrors_worst = sum(bitsAll(:) ~= rxBitsAll_worst(:));
    numErrorsMod_worst = sum(bitsModAll(:) ~= rxBitsModAll_worst(:));
    numErrorsAntenna_worst = sum(bitsAntennaAll(:) ~= rxBitsAntennaAll_worst(:));


    numBits = length(bitsAll(:));
    numBitsMod = length(bitsModAll(:));
    numBitsAntenna = length(bitsAntennaAll(:));

    ber_best = numErrors_best / numBits;
    berMod_best = numErrorsMod_best / numBitsMod;
    berAntenna_best = numErrorsAntenna_best / numBitsAntenna; 

    ber_worst = numErrors_worst / numBits;
    berMod_worst = numErrorsMod_worst / numBitsMod;
    berAntenna_worst = numErrorsAntenna_worst / numBitsAntenna; 


    % fprintf('BER: %f\n', ber_best);
    % fprintf('BER_sym: %f\n', berMod_best);
    fprintf('Best BER_ind: %f\n', berAntenna_best);
    fprintf('Worst BER_ind: %f\n', berAntenna_worst);
    fprintf('Total Errors: %d out of %d bits\n', numErrors_best, numBits);
        
        
    % Open a file for writing
    % fileID = fopen('ber_MA.txt', 'a');
    % fprintf(fileID, '%f\n', berAntenna);
    % fclose(fileID);
end





%% Functions

function [best_H, rand_H, worst_H] = get_selected_channels(N, M, Lt, Lr, lambda, kappa, rowIndices_t, rowIndices_r, t_tilde_grid, r_tilde_grid, HcolIndices, allSymbols)

    max_metric = 0;         % largest decision metric
    best_H = [];
    
    greedy_iters = 1;

    min_metric = inf; % tracked to compare best positions with worst positions
    worst_H = [];

    decision_metric_arr = zeros(size(rowIndices_t, 1), size(rowIndices_r, 1));
    
    % AoD for channel generation
    theta_t = random('Uniform', 0, pi, Lt, 1); % Lt x 1 array of elevation AoD for each transmit path
    phi_t = random('Uniform', 0, pi, Lt, 1);% Lt x 1 array of azimuth AoD for each transmit path
    
    % AoA for channel generation
    theta_r = random('Uniform', 0, pi, Lr, 1); % Lr x 1 array of elevation AoA for each receive path
    phi_r = random('Uniform', 0, pi, Lr, 1);% Lr x 1 array of azimuth AoA for each receive path
    
    % Path response matrix calculation shape Lr x Lt
    % (from origin of T region to origin of R region)
    var11 = kappa/(kappa + 1);
    sigma_path_response(1, 1) = sqrt(var11)*(randn(1) + 1i*randn(1));
    varpp = 1/((kappa+1)*(Lr-1));
    for p=2:Lr
        sigma_path_response(p, p) = sqrt(varpp)*(randn(1) + 1i*randn(1));
    end
    
    % Initialize random positions
    init_t_row_indices = rowIndices_t(randi(size(rowIndices_t, 1)), :);
    init_r_row_indices = rowIndices_r(randi(size(rowIndices_r, 1)), :);
    init_t_positions = t_tilde_grid(init_t_row_indices, :).';
    init_r_positions = r_tilde_grid(init_r_row_indices, :).'; % 2 x M RX positions (random)
    rand_H = get_channel(N, M, init_t_positions, init_r_positions, Lt, Lr, lambda, kappa, theta_t, phi_t, theta_r, phi_r, sigma_path_response);

    selected_t_positions = t_tilde_grid(init_t_row_indices, :).'; % 2 x N TX positions (random)
    selected_r_positions = r_tilde_grid(init_r_row_indices, :).'; % 2 x M RX positions (random)
    selected_t_indices = init_t_row_indices;
    selected_r_indices = init_r_row_indices;

    num_positions = size(t_tilde_grid, 1);
    positions_vec = (1:num_positions).';
    antenna_indices_t = 1:N;
    
    for optim_iter = 1:greedy_iters
    
        % Part 1: Optimize TX antennas
        for n = 1:N
            occupiedPositions = selected_t_indices([1:n-1 n+1:end]); % remove moving antenna
            availablePositions = positions_vec(~ismember(positions_vec, occupiedPositions)); % remove fixed pos. antenna positions
            temp_t_indices = selected_t_indices;
            for i = 1:size(availablePositions, 1)
                temp_t_indices(n) = availablePositions(i);
                temp_t_positions = t_tilde_grid(temp_t_indices, :).';    % 2 x N TX positions
                H = get_channel(N, M, temp_t_positions, selected_r_positions, Lt, Lr, lambda, kappa, theta_t, phi_t, theta_r, phi_r, sigma_path_response);
                H_dist_arr = zeros(size(HcolIndices, 1), length(allSymbols)^2);

                for k = 1:size(HcolIndices, 1)
                    index = 1;
                    for s1 = 1:length(allSymbols)
                        for s2 = 1:length(allSymbols)
                            % Calculate the norm between combinations of h1*s1 and h2*s2
                            H_dist_arr(k, index) = norm((H(:, HcolIndices(k, 1)) * allSymbols(s1)) - (H(:, HcolIndices(k, 2)) * allSymbols(s2)));
                            index = index + 1;
                        end
                    end
                end
                decision_metric = min(H_dist_arr(H_dist_arr ~= 0));
                
               
                % Track best channel
                if decision_metric > max_metric
                    max_metric = decision_metric;
                    best_H = H;
                    selected_t_positions = temp_t_positions; % Update the positions
                    selected_t_indices = temp_t_indices;
                end
                
            end
        end
        
    
        % Part 2: Optimize RX antennas
        for m = 1:M
            occupiedPositions = selected_r_indices([1:m-1 m+1:end]); % remove moving antenna
            availablePositions = positions_vec(~ismember(positions_vec, occupiedPositions)); % remove fixed pos. antenna positions
            temp_r_indices = selected_r_indices;
            for i = 1:size(availablePositions, 1)
                temp_r_indices(m) = availablePositions(i);
                temp_r_positions = r_tilde_grid(temp_r_indices, :).';    % 2 x N TX positions
                H = get_channel(N, M, selected_t_positions, temp_r_positions, Lt, Lr, lambda, kappa, theta_t, phi_t, theta_r, phi_r, sigma_path_response);
                H_dist_arr = zeros(size(HcolIndices, 1), length(allSymbols)^2);

                for k = 1:size(HcolIndices, 1)
                    index = 1;
                    for s1 = 1:length(allSymbols)
                        for s2 = 1:length(allSymbols)
                            % Calculate the norm between combinations of h1*s1 and h2*s2
                            H_dist_arr(k, index) = norm((H(:, HcolIndices(k, 1)) * allSymbols(s1)) - (H(:, HcolIndices(k, 2)) * allSymbols(s2)));
                            index = index + 1;
                        end
                    end
                end
    
                decision_metric = min(H_dist_arr(H_dist_arr ~= 0));
                
               
                % Track best channel
                if decision_metric > max_metric
                    max_metric = decision_metric;
                    best_H = H;
                    selected_r_positions = temp_r_positions; % Update the positions
                    selected_r_indices = temp_r_indices;
                end
                
            end
        end

    end
    

    worst_H = rand_H;
    % % Iterate over all combinations of TX positions and RX positions to get
    % % worst
    % for i = 1:size(rowIndices_t, 1)
    %     for j = 1:size(rowIndices_r, 1)
    %         selected_t_positions = t_tilde_grid(rowIndices_t(i, :), :).';    % 2 x N TX positions
    %         selected_r_positions = r_tilde_grid(rowIndices_r(j, :), :).';    % 2 x M RX positions
    %         H = get_channel(N, M, selected_t_positions, selected_r_positions, Lt, Lr, lambda, kappa, theta_t, phi_t, theta_r, phi_r, sigma_path_response);
    %         H_dist_arr = zeros(size(HcolIndices, 1), 1);
    % 
    %         for n = 1:size(HcolIndices, 1)
    %             H_dist_arr(n) = norm( H(:,HcolIndices(n,2)) - H(:,HcolIndices(n,1)) );
    %         end
    % 
    %         decision_metric = min(H_dist_arr);
    % 
    %         % Track worst channel
    %         if decision_metric < min_metric && decision_metric > 0
    %             min_metric = decision_metric;
    %             worst_H = H;
    %         end
    %     end
    % end

end

function H = get_channel(N, M, t_tilde, r_tilde, Lt, Lr, lambda, kappa, theta_t, phi_t, theta_r, phi_r, sigma_path_response)
    % CONSTANT AOD AND AOA CHANNEL
    % Far-field wireless channel model is considered.
    % Thus, for each channel path component, all MAs in the
    % transmit/receive region experience
    % the same angle of departure (AoD)/angle of arrival (AoA),
    % and amplitude of the complex path coefficient, while the
    % phase of the complex path coefficient varies for different
    % transmit/receive MAs at different positions.
    
    % Transmit MA field response matrix calculation shape: Lt x N
    
    
    rho_t = zeros(Lt, N);
    phase_diff_t = zeros(Lt, N);
    for t=1:N
        for p=1:Lt
            % cartesian coordinates of position t of current transmit MA
            x_t = t_tilde(1, t);
            y_t = t_tilde(2, t);
    
            % elevation and azimuth AoD of transmit path p
            thetaP_t = theta_t(p);
            phiP_t = phi_t(p);
    
            % diff of signal propagation for pth transmit path between position t and
            % origin of transmit region o_t (0,0,0)
            rho_t(p, t) = x_t * sin(thetaP_t) * cos(phiP_t) + y_t * cos(thetaP_t);
            phase_diff_t(p, t) = 2 * pi * rho_t(p, t)/lambda;
            
            % field response vector shape: 1 x Lt
            g(p) = exp(1i * phase_diff_t(p, t));
        end
    
        % field response matrix 
        G(:, t) = g.';
    end

    % Receive MA field response matrix calculation shape: Lt x N
    
    rho_r = zeros(Lr, M);
    phase_diff_r = zeros(Lr, M);
    for r=1:M
        for q=1:Lr
            % cartesian coordinates of position r of current receive MA
            x_r = r_tilde(1, r);
            y_r = r_tilde(2, r);  
    
            % elevation and azimuth AoA of receive path q
            thetaQ_r = theta_r(q);
            phiQ_r = phi_r(q);
    
            % diff of signal propagation for qth receive path between position r and
            % origin of receive region o_r (0,0,0)
            rho_r(q, r) = x_r*sin(thetaQ_r)*cos(phiQ_r) + y_r*cos(thetaQ_r);
            phase_diff_r(q, r) = 2*pi*rho_r(q, r)/lambda;
    
            % field response vector shape: 1 x Lr
            f(q) = exp(1i * phase_diff_r(q, r));
        end
        % field response matrix 
        F(:, r) = f.';
    end
    
    
    
    % Final channel matrix
    H = F' * sigma_path_response * G;
   
end

function [rxBits, rxBitsMod, rxBitsAntenna] = ML_detect(rxSignal, H, numTx, modOrder, allSymbols, allBitsMod)
    minDist = inf;
    bestIndex = 0;
    bestSymbolIndex = 0;
    for txIdx = 1:numTx
        for symIdx = 1:modOrder
            testSignal = zeros(1, numTx);
            testSignal(txIdx) = allSymbols(symIdx);
            estimatedRxSignal = H * testSignal.';
            dist = norm(rxSignal - estimatedRxSignal);
            if dist < minDist
                minDist = dist;
                bestIndex = txIdx;
                bestSymbolIndex = symIdx;
            end
        end
    end

    % Get the bits back from the detected index and symbol
    rxBitsMod = allBitsMod(bestSymbolIndex, :);
    rxBitsAntenna = de2bi(bestIndex-1, log2(numTx), 'left-msb');
    rxBits = [rxBitsAntenna, rxBitsMod];
end


function plotBERvsSNR()
    
    numTx = 4; % num transmit MAs
    numRx = 4; % num receive MAs
    
    channel_config.Lt = 3; % num transmit paths
    channel_config.Lr = 3; % num receive paths 
    
    M_sym = 16;
    modOrder = M_sym;
    m_sym = log2(M_sym);
    
    avg_noise_pow = 1;
    sigma = sqrt(avg_noise_pow);    % std of noise
    
    fc = 2.4e9;         % carrier freq
    lambda = 3e8 / fc; % carrier wavelength
    A = lambda;       % TX MAs and RX MAs can move in a square region of AxA
    dist_between_positions = (lambda/2);
    kappa = 1;  % kappa parameter for field-response based channel model
    channel_config.kappa = kappa;
    channel_config.lambda = lambda;
    
    % Generate transmit MA and receive MA position grids (each row is a possible location from 0 to 4*lambda with lambda/2 spacing)
    %coordinates = 0 : dist_between_positions : A;
    coordinates = -1*dist_between_positions : dist_between_positions : 1*dist_between_positions;
    [x_grid, y_grid] = ndgrid(coordinates, coordinates);
    channel_config.t_tilde_grid = [x_grid(:), y_grid(:)];   % num_t_positions x 2 TX positions
    channel_config.r_tilde_grid = [x_grid(:), y_grid(:)];   % num_r_positions x 2 RX positions
    
    % % Visualize the grid of positions for TX antennas
    % figure;
    % scatter(t_tilde_grid(:, 1), t_tilde_grid(:, 2));
    % title('TX position grid')
    
    % Generate all possible row selections from TX and RX grids (N TX antennas and M RX antennas)
    channel_config.rowIndices_t = nchoosek(1:size(channel_config.t_tilde_grid, 1), numTx);
    channel_config.rowIndices_r = nchoosek(1:size(channel_config.r_tilde_grid, 1), numRx);
    
    % Generate all possible column pair combinations from H
    HcolIndices = nchoosek(1:numTx, 2);
    diagonalPairs = [(1:numTx)' (1:numTx)']; % Adding pairs like [1 1], [2 2], [3 3], [4 4]
    HcolIndices = [diagonalPairs; HcolIndices];
    channel_config.HcolIndices = HcolIndices;
    % Generate all possible symbols
    ss = qammod(0 : M_sym-1, M_sym, "Gray", "UnitAveragePower", true);
    
    % Generate all possible antenna selection vectors (this is for SM - update this for other cases)
    e = eye(numTx);
    
    snrRange = 0:4:20;      % SNR range from 0 dB to 30 dB

    % Initialize BER storage
    berResults_best = zeros(1, length(snrRange));
    berResults_worst = zeros(1, length(snrRange));
    
    berModResults_best = zeros(1, length(snrRange));
    berModResults_worst = zeros(1, length(snrRange));

    berAntennaResults_best = zeros(1, length(snrRange));
    berAntennaResults_worst = zeros(1, length(snrRange));
    
    

    % Loop over SNR values
    for i = 1:length(snrRange)
        snrDb = snrRange(i);
        fprintf('Simulating SNR = %d dB...\n', snrDb);
        if snrDb <= 7
            num_iters = 5e3;      % Number of iterations per SNR point
        elseif 7 < snrDb <=14
            num_iters = 5e4;
        else
            num_iters = 8e7;
        end
        [berResults_best(i), berModResults_best(i), berAntennaResults_best(i) ...
        , berResults_worst(i), berModResults_worst(i), berAntennaResults_worst(i) ...
        , ~, ~] = simulateSpatialModulationRayleighML(numTx, numRx, modOrder ...
        , num_iters, snrDb, channel_config);
    end
    results.berResults_best = berResults_best;
    results.berResults_worst = berResults_worst;

    results.berModResults_best = berModResults_best;
    results.berModResults_worst = berModResults_worst;

    results.berAntennaResults_best = berAntennaResults_best;
    results.berAntennaResults_worst = berAntennaResults_worst;
    
    save("results.mat", "results");

    % Plot BER results
    figure;
    semilogy(snrRange, berResults_best, '-o');
    hold on;
    semilogy(snrRange, berResults_worst, '--');
    grid on;
    xlabel('SNR (dB)');
    ylabel('BER');
    title('BER vs SNR');
    legend("Greedy", "Worst")

    % Plot Modulation BER results
    figure;
    semilogy(snrRange, berModResults_best, '-o');
    hold on;
    semilogy(snrRange, berModResults_worst, '--');
    grid on;
    xlabel('SNR (dB)');
    ylabel('Modulation BER');
    title('Modulation BER vs SNR');
    legend("Greedy", "Worst")

    % Plot Modulation BER results
    figure;
    semilogy(snrRange, berAntennaResults_best, '-o');
    hold on;
    semilogy(snrRange, berAntennaResults_worst, '--');
    grid on;
    xlabel('SNR (dB)');
    ylabel('Index BER');
    title('Index BER vs SNR');
    legend("Greedy", "Worst")
end
