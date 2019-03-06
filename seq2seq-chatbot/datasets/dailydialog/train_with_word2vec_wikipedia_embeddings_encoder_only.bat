call C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3
cd ..\..
python train.py --datasetdir=datasets\dailydialog --encoderembeddingsdir=embeddings\word2vec_wikipedia

cmd /k