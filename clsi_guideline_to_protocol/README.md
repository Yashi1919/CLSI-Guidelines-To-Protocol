#Create a python virtual environment

python -m venv venv

activate it

.\venv\scripts\activate


#Create a new directory

mkdir clsi_protocol_to_guideline
cd clsi_protocol_to_guideline


#clone the repo

git clone https://github.com/Yashi1919/CLSI-Guidelines-To-Protocol.git

cd clsi_guideline_to_protocol


#Download the requirements

pip install -r requirements.txt

#Set Environment variables

Gemini api key : https://aistudio.google.com/apikey
Serper api key : https://serper.dev/api-key

place those keys inside strings in the .env file

you can change model by going to line 19 in main.py

#place the pdf in the directory

place the pdf path in line 376





