virtualenv --python=python3 .env

cat environmental_vars.txt >> .env/bin/activate

source .env/bin/activate

pip install -r requirements.txt
