%% comparar_modelos_rmse.m
% Compara salidas de distintos modelos usando RMSE

clear; clc;

% Cargar datos reales
load datos_reales.mat   % debe contener y_real (vector)

% Cargar salidas de Simulink para cada modelo
% Ajustá los nombres de archivo / variables según tu setup

load salida_net_20.mat   % y_hat_20
load salida_net_10.mat   % y_hat_10
load salida_net_5.mat    % y_hat_5
load salida_net_2.mat    % y_hat_2
load salida_net_lin.mat  % y_hat_lin  (perceptrón simple / red lineal)
load salida_modelo_lin.mat % y_hat_modelo_lin (tu modelo lineal aparte)

% Asegurar vectores columna
y = y_real(:);

y20   = y_hat_20(:);
y10   = y_hat_10(:);
y5    = y_hat_5(:);
y2    = y_hat_2(:);
ylin  = y_hat_lin(:);
yml   = y_hat_modelo_lin(:);

% Ajustar longitud mínima común (por seguridad)
L = min([length(y), length(y20), length(y10), length(y5), ...
         length(y2), length(ylin), length(yml)]);

y    = y(1:L);
y20  = y20(1:L);
y10  = y10(1:L);
y5   = y5(1:L);
y2   = y2(1:L);
ylin = ylin(1:L);
yml  = yml(1:L);

% Calcular RMSE para cada modelo
rmse_20  = rmse(y, y20);
rmse_10  = rmse(y, y10);
rmse_5   = rmse(y, y5);
rmse_2   = rmse(y, y2);
rmse_lin = rmse(y, ylin);
rmse_ml  = rmse(y, yml);

% Mostrar resultados en consola
fprintf('RMSE MLP 20 neuronas:      %.6f\n', rmse_20);
fprintf('RMSE MLP 10 neuronas:      %.6f\n', rmse_10);
fprintf('RMSE MLP 5 neuronas:       %.6f\n', rmse_5);
fprintf('RMSE MLP 2 neuronas:       %.6f\n', rmse_2);
fprintf('RMSE perceptron lineal:    %.6f\n', rmse_lin);
fprintf('RMSE modelo lineal (PL):   %.6f\n', rmse_ml);

% Opcional: guardar en un .mat o .csv para llevar a LaTeX
resultados_nombres = { ...
    'MLP 20', ...
    'MLP 10', ...
    'MLP 5',  ...
    'MLP 2',  ...
    'Perceptron lineal', ...
    'Modelo lineal'};

resultados_rmse = [rmse_20; rmse_10; rmse_5; rmse_2; rmse_lin; rmse_ml];

save resultados_rmse.mat resultados_nombres resultados_rmse;

% También podés generar algo tipo:
% writematrix(resultados_rmse, 'resultados_rmse.csv');
