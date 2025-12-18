function net = entrenar_red(Xn, Tn, numNeuronas)
% ENTRENAR_RED Entrena una red neuronal feedforward (fitnet)
%
% Uso:
%    net = entrenar_red(Xn, Tn, numNeuronas)
%
% Entradas:
%    Xn: matriz de entrada normalizada (entradas × muestras)
%    Tn: matriz de salida normalizada (salidas × muestras)
%    numNeuronas: cantidad de neuronas en capa oculta
%
% Salidas:
%    net: red neuronal entrenada

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

    % Entrenamiento
    net = train(net, Xn.', Tn.');
end

% =============================================================
function guardar_red(nombreArchivo, net, muX, sigX, muT, sigT, ny, nu)
% GUARDAR_RED Guarda la red entrenada junto con parámetros
    save(nombreArchivo, 'net', 'muX', 'sigX', 'muT', 'sigT', 'ny', 'nu');
end

% =============================================================
function net = cargar_red(nombreArchivo)
% CARGAR_RED Carga una red y la manda al workspace
    data = load(nombreArchivo);
    net = data.net;
    assignin('base', 'net', net);
    gensim(net);
end

% =============================================================
function net = entrenar_perceptron_simple(Xn, Tn)
% ENTRENAR_PERCEPTRON_SIMPLE Entrena un perceptrón simple
%
% Uso:
%    net = entrenar_perceptron_simple(Xn, Tn)

    P = Xn;
    T = Tn;
    rangoEntradas = [min(P, [], 2) max(P, [], 2)];

    % Crear perceptrón lineal
    net = newff(rangoEntradas, size(T,1), {'purelin'}, 'trainlm');

    % Desactivar preprocesamiento
    net.inputs{1}.processFcns = {};
    net.outputs{1}.processFcns = {};

    % Entrenar
    net = train(net, P, T);

    % Visualizar la red en Simulink
    gensim(net);
end
