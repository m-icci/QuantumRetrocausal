# KuCoin
$env:KUCOIN_API_KEY = '67d43c74df512e0001bc6867'
$env:KUCOIN_SECRET_KEY = '51da51df-644f-457a-88cf-be2ea59b565f'
$env:KUCOIN_PASSPHRASE = 'mcs311205'

# Kraken
$env:KRAKEN_API_KEY = 'Q5Bo76UH0RWwamNmPD0u8U73yh0lv2uMZQDH6+jXV6QL2SRpedfxoRbn'
$env:KRAKEN_API_SECRET = 'dBwcdsPfgtpGspL6cpN49jYdCdflUvLSTKXLr8M9tEqaGvJY7tdHp/kPM59GDgu3vX/fKdtHxIw+a0+fTf+jMA=='

Write-Host "Variáveis de ambiente configuradas com sucesso!"
Get-ChildItem env: | Where-Object { $_.Name -like '*KUCOIN*' -or $_.Name -like '*KRAKEN*' } | Format-List 