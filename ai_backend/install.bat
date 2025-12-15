pip install -r requirements.txt
pip install -r requirements.txt --upgrade

pip-compile requirements.txt --output-file requirements.lock

pip install -r requirements.lock --upgrade