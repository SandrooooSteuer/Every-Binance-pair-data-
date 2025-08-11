# Every-Binance-pair-data-
You can change the code to every Pair and Timeframe you want. 
With RSI 14 30 50 and EMA 13 50 100 200 800 2000. Includet 
The data file will be auto created.

This PACKS you need Windows Terminal: # In ein leeres Projektverzeichnis gehen
mkdir btc_loader && cd btc_loader


py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

ni requirements.txt -Force

pip install -r requirements.txt

For mac/Linux mkdir -p btc_loader && cd btc_loader
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
touch requirements.txt
pip install -r requirements.txt
