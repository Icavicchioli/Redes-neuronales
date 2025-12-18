function net = entrenar_red(Xn, Tn, numNeuronas)
% ENTRENAR_RED Entrena una red neuronal feedforward (fitnet)
%
% Uso:
%   net = entrenar_red(Xn, Tn, numNeuronas)
%
% Entradas:
%   Xn: matriz de entrada (muestras × entradas)
%   Tn: matriz de salida  (muestras × salidas)
%   numNeuronas: cantidad de neuronas en capa oculta
%
% Salida:
%   net: red neuronal entrenada

    % Crear red
    net = fitnet(numNeuronas);

    % Configurar funciones de transferencia
    net.layers{1}.transferFcn = 'tansig';
    net.layers{2}.transferFcn = 'purelin';

    % División de datos
    net.divideFcn = 'dividerand';
    net.divideParam.trainRatio = 0.7;
    net.divideParam.valRatio   = 0.15;
    net.divideParam.testRatio  = 0.15;

    % Desactivar preprocesamiento interno
    net.inputs{1}.processFcns  = {};
    net.outputs{2}.processFcns = {};

    % Entrenamiento (columnas = muestras)
    net = train(net, Xn.', Tn.');
end

% =============================================================
function guardar_red(nombreArchivo, net, muX, sigX, muT, sigT, ny, nu)
% GUARDAR_RED Guarda la red entrenada junto con parámetros
    save(nombreArchivo, 'net', 'muX', 'sigX', 'muT', 'sigT', 'ny', 'nu');
end

% =============================================================
function net = cargar_red(nombreArchivo)
% CARGAR_RED Carga una red y la manda al workspace y Simulink
    data = load(nombreArchivo);
    net = data.net;
    assignin('base', 'net', net);
    gensim(net);
end

% =============================================================
function net = entrenar_perceptron_simple(Xn, Tn)
% ENTRENAR_PERCEPTRON_SIMPLE Entrena un perceptrón lineal (sin capa oculta)
%
% Uso:
%   net = entrenar_perceptron_simple(Xn, Tn)
%
% Entradas:
%   Xn: matriz de entrada (muestras × entradas)
%   Tn: matriz de salida  (muestras × salidas)

    P = Xn;   % muestras × entradas
    T = Tn;   % muestras × salidas

    % Rango de entrada (por característica)
    rangoEntradas = [min(P, [], 1).'  max(P, [], 1).'];

    % Red lineal pura (sin capa oculta)
    net = feedforwardnet([]);        % equivalente a capa lineal
    net.layers{1}.transferFcn = 'purelin';

    % Desactivar preprocesamiento
    net.inputs{1}.processFcns  = {};
    net.outputs{1}.processFcns = {};

    % Entrenamiento (columnas = muestras)
    net = train(net, P.', T.');

    % Opcional: visualizar en Simulink
    gensim(net);
end

% =============================================================

function e = rmse(y, yhat)
% RMSE Calcula el error cuadrático medio (root mean square error)
%
% Uso:
%   e = rmse(y, yhat)
%
% Entradas:
%   y    : secuencia real (vector)
%   yhat : secuencia estimada (vector)
%
% Salida:
%   e    : valor escalar del RMSE

    % Asegurar vectores columna
    y    = y(:);
    yhat = yhat(:);

    if length(y) ~= length(yhat)
        error('rmse: Las secuencias deben tener la misma longitud.');
    end

    e = sqrt(mean((y - yhat).^2));
end

% =============================================================



function [Xn, Tn, muX, sigX, muT, sigT, ny, nu] = preparar_datos(out, ny, nu)
    % Extraer y, u
    y = out.Y(:);
    u = out.u(:);

    N  = length(y);
    k0 = max(ny, nu);
    Ns = N - k0 - 1;

    X = zeros(Ns, ny + nu);
    T = zeros(Ns, 1);

    for k = 1:Ns
        t = k + k0;
        X(k, 1:ny)     = y(t:-1:t-ny+1);
        X(k, ny+1:end) = u(t:-1:t-nu+1);
        T(k)           = y(t+1);
    end

    % Normalización
    muX = mean(X, 1);
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
% =============================================================
