$ErrorActionPreference = 'Stop'

Write-Host 'Verificando o Python 3.13...'
py -3.13 --version

Write-Host 'Criando o ambiente virtual em .venv...'
py -3.13 -m venv .venv

$pythonProjeto = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

Write-Host 'Atualizando o pip...'
& $pythonProjeto -m pip install --upgrade pip

Write-Host 'Instalando as dependencias do projeto...'
& $pythonProjeto -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')

Write-Host ''
Write-Host 'Ambiente configurado com sucesso.'
Write-Host 'No VS Code, selecione o kernel Python localizado em .venv.'
