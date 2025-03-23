@echo off
echo ===== Iniciando QUALIA em modo real =====
echo.

echo Verificando ambiente Python...
python --version

echo.
echo Configurando ambiente...
set PYTHONPATH=%CD%

echo.
echo Executando sistema de trading em modo real...
python quantum_trading/run_integrated_scalping.py --mode real --debug

echo.
echo ===== Fim da execucao =====
pause 