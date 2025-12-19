function net = cargar_red(nombreArchivo)
% CARGAR_RED Carga una red y la manda al workspace y Simulink
    data = load(nombreArchivo);
    net = data.net;
    assignin('base', 'net', net);
    gensim(net);
end