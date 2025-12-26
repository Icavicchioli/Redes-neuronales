%% ============================
%  ARMADO DE DATASET
% =============================
load('clon_controlador_dataset.mat')
%%

% Extraer señales desde Simulink
y = out.Y(:);        % salida de la planta
r = out.ref(:);      % referencia
e = out.error(:);    % error
c = out.C;
N = length(y);

%% Parámetros del modelo
na = 10;                 % retardos de salida
n0 = na + 1;
Ns = N - na;

% Inicialización
X = zeros(na + 2, Ns);   % [y(k-1..k-10); r(k); e(k)]
T = zeros(1, Ns);        % y(k)

for k = n0:N
    X(1:na, k-na) = y(k-1:-1:k-na);  % salidas pasadas
    X(na+1, k-na) = r(k);            % referencia actual
    X(na+2, k-na) = e(k);            % error actual
    T(k-na)       = c(k);            % salida objetivo
end

%% ============================
%  DIVISIÓN TRAIN / TEST
% ============================

trainRatio = 0.70;
Ntrain = floor(trainRatio * Ns);

idxTrain = 1:Ntrain;
idxTest  = (Ntrain+1):Ns;

%% ============================
%  CREACIÓN DE LA RED
% ============================

numNeuronas = 20;
net = fitnet(numNeuronas, 'trainlm');

% Funciones de activación
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';

% Función de costo
net.performFcn = 'mse';

% División explícita SIN validación
net.divideFcn = 'divideind';
net.divideParam.trainInd = idxTrain;
net.divideParam.valInd   = [];
net.divideParam.testInd  = idxTest;

%% ============================
%  PARÁMETROS DE ENTRENAMIENTO
% ============================

epochs = 1000;

net.trainParam.epochs   = epochs;
net.trainParam.min_grad = 1e-9;
net.trainParam.mu_max   = 1e12;
net.trainParam.time     = inf;

%% ============================
%  ENTRENAMIENTO
% ============================

[net, tr] = train(net, X, T);

%% ============================
%  EVALUACIÓN BÁSICA
% ============================

Yhat = net(X);

figure;
subplot(2,1,1)
plot(T, 'k'); hold on;
plot(Yhat, 'r--');
grid on;
legend('Planta real', 'Red neuronal');
title('Salida real vs NN');

subplot(2,1,2)
plot(T - Yhat);
grid on;
title('Error de predicción');

%% ============================

gensim(net);

%% otra sin referencia

%% ============================
%  ARMADO DE DATASET
% ============================
clc;
clear all;
load('clon_controlador_dataset_2.mat')

% Extraer señales desde Simulink
y = out.Y(:);        % salida de la planta
e = out.error(:);    % error
c = out.C(:);        % acción de control (PID)

N = length(y);

%% Parámetros del modelo
na = 10;                 % retardos de salida
n0 = na + 1;
Ns = N - na;

% Inicialización
X = zeros(na + 1, Ns);   % [y(k-1..k-10); e(k)]
T = zeros(1, Ns);        % C(k)

for k = n0:N
    X(1:na, k-na) = y(k-1:-1:k-na);  % salidas pasadas
    X(na+1, k-na) = e(k);            % error actual
    T(k-na)       = c(k);            % salida objetivo
end

%% ============================
%  DIVISIÓN TRAIN / TEST
% ============================

trainRatio = 0.70;
Ntrain = floor(trainRatio * Ns);

idxTrain = 1:Ntrain;
idxTest  = (Ntrain+1):Ns;

%% ============================
%  CREACIÓN DE LA RED
% ============================

numNeuronas = 20;
net = fitnet(numNeuronas, 'trainlm');

% Funciones de activación
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin';

% Función de costo
net.performFcn = 'mse';

% División explícita SIN validación
net.divideFcn = 'divideind';
net.divideParam.trainInd = idxTrain;
net.divideParam.valInd   = [];
net.divideParam.testInd  = idxTest;

%% ============================
%  PARÁMETROS DE ENTRENAMIENTO
% ============================

epochs = 2000;

net.trainParam.epochs   = epochs;
net.trainParam.min_grad = 1e-9;
net.trainParam.mu_max   = 1e12;
net.trainParam.time     = inf;

%% ============================
%  ENTRENAMIENTO
% ============================

[net, tr] = train(net, X, T);


%% ============================
%  GENERAR BLOQUE SIMULINK
% ============================

gensim(net);
