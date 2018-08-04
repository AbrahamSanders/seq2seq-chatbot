call C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3
cd ..\..
python train.py --datasetdir=datasets\cornell_movie_dialog --encoderembeddingsdir=embeddings\dependency_based

cmd /k