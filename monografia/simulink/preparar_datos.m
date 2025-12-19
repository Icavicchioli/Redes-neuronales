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