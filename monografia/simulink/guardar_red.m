function guardar_red(nombreArchivo, net, muX, sigX, muT, sigT, ny, nu)
% GUARDAR_RED Guarda la red entrenada junto con parámetros
    save(nombreArchivo, 'net', 'muX', 'sigX', 'muT', 'sigT', 'ny', 'nu');
end

