# Запуск
Python 3.12.4

git clone https://github.com/ria-com/nomeroff-net.git
cd nomeroff-net

python -m venv venv
venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install "setuptools>=60.0.0,<70.0.0" wheel

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -m pip install -r requirements.txt
python -m pip install -r requirements-api.txt

python test.py
python main.py

