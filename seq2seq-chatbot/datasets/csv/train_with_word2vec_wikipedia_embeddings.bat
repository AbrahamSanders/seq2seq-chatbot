call C:\ProgramData\Anaconda3\Scripts\activate.bat C:\ProgramData\Anaconda3
cd ..\..
python train.py --datasetdir=datasets\csv --encoderembeddingsdir=embeddings\word2vec_wikipedia --decoderembeddingsdir=embeddings\word2vec_wikipedia

cmd /k