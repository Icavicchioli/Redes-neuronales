%% entrenar_con_datasets.m
% Entrena redes para dataset14400 y dataset3600
% Version con experimentos explícitos (sin loop)

clear; clc;

ny = 10;
nu = 1;

% DATASET 14400
%% ============================================================

archivo = 'dataset14400.mat';
fprintf('\n=== Procesando %s ===\n', archivo);

% Cargar dataset
load(archivo, 'out');

% Preparar datos
[Xn, Tn, muX, sigX, muT, sigT, ny, nu] = preparar_datos(out, ny, nu);

%% --- MLP 20 neuronas ---
net20 = entrenar_red(Xn, Tn, 20,2000);
guardar_red('net_20_dataset14400.mat', ...
            net20, muX, sigX, muT, sigT, ny, nu);

%% --- MLP 10 neuronas ---
net10 = entrenar_red(Xn, Tn, 10,1000);
guardar_red('net_10_dataset14400.mat', ...
            net10, muX, sigX, muT, sigT, ny, nu);

%% --- MLP 5 neuronas ---
net5 = entrenar_red(Xn, Tn, 5,1000);
guardar_red('net_5_dataset14400.mat', ...
            net5, muX, sigX, muT, sigT, ny, nu);

%% --- MLP 2 neuronas ---
net2 = entrenar_red(Xn, Tn, 2,4000);
guardar_red('net_2_dataset14400.mat', ...
            net2, muX, sigX, muT, sigT, ny, nu);



% DATASET 3600
%% ============================================================

archivo = 'dataset3600.mat';
fprintf('\n=== Procesando %s ===\n', archivo);

% Cargar dataset
load(archivo, 'out');

% Preparar datos
[Xn, Tn, muX, sigX, muT, sigT, ny, nu] = preparar_datos(out, ny, nu);

%% --- MLP 20 neuronas ---
net20 = entrenar_red(Xn, Tn, 20,2000);
guardar_red('net_20_dataset3600.mat', ...
            net20, muX, sigX, muT, sigT, ny, nu);

%% --- MLP 10 neuronas ---
net10 = entrenar_red(Xn, Tn, 10,1000);
guardar_red('net_10_dataset3600.mat', ...
            net10, muX, sigX, muT, sigT, ny, nu);

%% --- MLP 5 neuronas ---
net5 = entrenar_red(Xn, Tn, 5,1000);
guardar_red('net_5_dataset3600.mat', ...
            net5, muX, sigX, muT, sigT, ny, nu);

%% --- MLP 2 neuronas ---
net2 = entrenar_red(Xn, Tn, 2,1000);
guardar_red('net_2_dataset3600.mat', ...
            net2, muX, sigX, muT, sigT, ny, nu);


