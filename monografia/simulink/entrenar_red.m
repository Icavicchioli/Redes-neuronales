function net = entrenar_red(Xn, Tn, numNeuronas, epochs)
% ENTRENAR_RED Entrena una red neuronal feedforward (fitnet)
% sin early stopping y con división determinista
%
% Uso:
%   net = entrenar_red(Xn, Tn, numNeuronas)

    % ------------------------------------------------------------
    % Crear red
    % ------------------------------------------------------------
    net = fitnet(numNeuronas);

    % Funciones de transferencia
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';

    % ------------------------------------------------------------
    % División determinista SIN validación
    % ------------------------------------------------------------
    N = size(Xn, 1);   % muestras

    nTrain = floor(0.85 * N);

    idxTrain = 1 : nTrain;
    idxTest  = nTrain + 1 : N;

    net.divideFcn = 'divideind';
    net.divideParam.trainInd = idxTrain;
    net.divideParam.valInd   = [];      % <-- elimina early stopping
    net.divideParam.testInd  = idxTest;
    
    net.trainParam.epochs = epochs;
    net.trainParam.min_grad = 1e-9;
    net.trainParam.mu_max = 1e12;
    net.trainParam.time = inf;


    % ------------------------------------------------------------
    % Desactivar preprocesamiento interno
    % ------------------------------------------------------------
    net.inputs{1}.processFcns  = {};
    net.outputs{2}.processFcns = {};

    % ------------------------------------------------------------
    % Entrenamiento
    % ------------------------------------------------------------
    net = train(net, Xn.', Tn.');
end
