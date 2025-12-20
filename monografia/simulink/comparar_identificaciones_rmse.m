%% RMSE sin transitorio (primeras 1000 muestras)
% Todo explícito, señales desde out

k0 = 2000;

% Señal real
y = out.Y(k0:end);

%% ---------------- LINEAL ----------------
rmse_Y_lin = rmse(y, out.Y_lin(k0:end))
rmse_Y_lin_est = rmse(y, out.Y_lin_est(k0:end))

%% ---------------- MLP (10 neuronas) ----------------
rmse_MLP1014400 = rmse(y, out.MLP1014400(k0:end))
rmse_MLP103600  = rmse(y, out.MLP103600(k0:end))

%% ---------------- MLP (20 neuronas) ----------------
rmse_MLP2014400 = rmse(y, out.MLP2014400(k0:end))
rmse_MLP203600  = rmse(y, out.MLP203600(k0:end))

%% ---------------- MLP (2 neuronas) ----------------
rmse_MLP214400 = rmse(y, out.MLP214400(k0:end))
rmse_MLP23600  = rmse(y, out.MLP23600(k0:end))

%% ---------------- MLP (5 neuronas) ----------------
rmse_MLP514400 = rmse(y, out.MLP514400(k0:end))
rmse_MLP53600  = rmse(y, out.MLP53600(k0:end))
