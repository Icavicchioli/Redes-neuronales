%% red_recurrente_LSTM.m
% Entrenamiento y evaluación de una red recurrente (LSTM)
% usando Deep Network Designer
%
% Requisitos:
% - out.u : entrada de la planta (vector)
% - out.Y : salida de la planta (vector)

%clear; 
clc;

%% ============================================================
% Cargar dataset
%% ============================================================

archivo = 'dataset14400.mat';
fprintf('\n=== Procesando %s ===\n', archivo);

load(archivo,'out');

u = out.u(:)';   % entrada (fila)
y = out.Y(:)';   % salida  (fila)

assert(length(u) == length(y),'u e y deben tener igual longitud');

N = length(u);

%% ============================================================
% Normalización (guardar parámetros)
%% ============================================================

mu_u  = mean(u);
sig_u = std(u);

mu_y  = mean(y);
sig_y = std(y);

u_n = (u - mu_u) / sig_u;
y_n = (y - mu_y) / sig_y;

%% ============================================================
% Armar secuencias para LSTM
%% ============================================================

T = size(u_n, 2);

fracTrain = 0.7;
Ttrain = floor(fracTrain * T);

% Train
u_train = u_n(:, 1:Ttrain);
y_train = y_n(:, 1:Ttrain);

% Test
u_test  = u_n(:, Ttrain+1:end);
y_test  = y_n(:, Ttrain+1:end);

dsXTrain = arrayDatastore(u_train, 'IterationDimension', 2);
dsYTrain = arrayDatastore(y_train, 'IterationDimension', 2);
dsTrain  = combine(dsXTrain, dsYTrain);

dsXTest = arrayDatastore(u_test, 'IterationDimension', 2);
dsYTest = arrayDatastore(y_test, 'IterationDimension', 2);
dsTest  = combine(dsXTest, dsYTest);

preview(dsTrain)
preview(dsTest)
%%



