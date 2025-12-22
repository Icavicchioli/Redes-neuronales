clear; clc;

ny = 8;
nu = 1;

% DATASET 14400con lineal
%% ============================================================

archivo = 'LINdataset14400.mat';
fprintf('\n=== Procesando %s ===\n', archivo);

% Cargar dataset
load(archivo, 'out');

% Preparar datos
[Xn, Tn, muX, sigX, muT, sigT, ny, nu] =  preparar_datos_lin2real(out.y_lin, out.Y, out.u, ny, nu);

%% --- MLP ---
net20 = entrenar_redV2(Xn, Tn, 5,1000);
guardar_red('LIN_prueba1.mat', net20, muX, sigX, muT, sigT, ny, nu);
cargar_red('LIN_prueba1.mat');

%% RMSE sin transitorio (primeras 1000 muestras)
% Todo explícito, señales desde out

k0 = 2000;
% Señal real

%% ---------------- LINEAL ----------------
rmse(out.Y(k0:end), out.y_est(k0:end))


%%
function [Xn, Tn, muX, sigX, muT, sigT, ny, nu] = ...
    preparar_datos_lin2real(y_lin, y_real, u, ny, nu)
% PREPARAR_DATOS_LIN2REAL
% Arma un dataset para entrenar una red que mapea
% (y_lin, u) -> error de estimación lineal y - y_lin
%
% Entradas:
%   y_lin  : salida del sistema lineal
%   y_real : salida real del sistema no lineal
%   u      : entrada del sistema
%   ny     : cantidad de retardos de y_lin
%   nu     : cantidad de retardos de u
%
% Salidas:
%   Xn, Tn : datos normalizados
%   mu*,sig* : parámetros de normalización

    y_lin  = y_lin(:);
    y_real = y_real(:);
    u      = u(:);

    N  = length(y_lin);
    k0 = max(ny, nu);
    Ns = N - k0;

    X = zeros(Ns, ny + nu);
    T = zeros(Ns, 1);

    for k = 1:Ns
        t = k + k0;

        % Retardos de la salida lineal
        X(k, 1:ny) = y_lin(t:-1:t-ny+1);

        % Retardos de la entrada
        X(k, ny+1:end) = u(t:-1:t-nu+1);

        % Target: salida real
        T(k) = y_real(t) - y_lin(t);


    end

    % Normalización
    muX  = mean(X, 1);
    sigX = std(X, 0, 1);
    sigX(sigX == 0) = 1;

    muT  = mean(T);
    sigT = std(T);
    if sigT == 0
        sigT = 1;
    end

    Xn = (X - muX) ./ sigX;
    Tn = (T - muT) ./ sigT;
end


function net = entrenar_redV2(Xn, Tn, numNeuronas, epochs)
% entrenar_redV2 Entrena una red neuronal feedforward (fitnet)
% sin early stopping y con división determinista
%
% Uso:
%   net = entrenar_redV2(Xn, Tn, numNeuronas)

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


    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio   = 0.15;
    net.divideParam.testRatio  = 0.15;

    
    net.trainParam.epochs = epochs;

 

    % ------------------------------------------------------------
    % Entrenamiento
    % ------------------------------------------------------------
    net = train(net, Xn.', Tn.');
end

