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