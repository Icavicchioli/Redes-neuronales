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