%% cargar_todas_las_redes.m
% Carga redes entrenadas y genera bloques Simulink (gensim)
% Ejecutar por secciones

clear; clc;

% DATASET 14400
%% --- MLP 20 neuronas | dataset14400 ---
cargar_red('net_20_dataset14400.mat');

%% --- MLP 10 neuronas | dataset14400 ---
cargar_red('net_10_dataset14400.mat');

%% --- MLP 5 neuronas | dataset14400 ---
cargar_red('net_5_dataset14400.mat');

%% --- MLP 2 neuronas | dataset14400 ---
cargar_red('net_2_dataset14400.mat');

%% --- Perceptrón lineal | dataset14400 ---
cargar_red('net_lin_dataset14400.mat');


% DATASET 3600
%% --- MLP 20 neuronas | dataset3600 ---
cargar_red('net_20_dataset3600.mat');

%% --- MLP 10 neuronas | dataset3600 ---
cargar_red('net_10_dataset3600.mat');

%% --- MLP 5 neuronas | dataset3600 ---
cargar_red('net_5_dataset3600.mat');

%% --- MLP 2 neuronas | dataset3600 ---
cargar_red('net_2_dataset3600.mat');

%% --- Perceptrón lineal | dataset3600 ---
cargar_red('net_lin_dataset3600.mat');
