$env:KRAKEN_API_KEY = 'Q5Bo76UH0RWwamNmPD0u8U73yh0lv2uMZQDH6+jXV6QL2SRpedfxoRbn'
$env:KRAKEN_API_SECRET = 'dBwcdsPfgtpGspL6cpN49jYdCdflUvLSTKXLr8M9tEqaGvJY7tdHp/kPM59GDgu3vX/fKdtHxIw+a0+fTf+jMA=='

Write-Host "Variáveis de ambiente da Kraken configuradas com sucesso!"
Get-ChildItem env: | Where-Object { $_.Name -like '*KRAKEN*' } | Format-List 