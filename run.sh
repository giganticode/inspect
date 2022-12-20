CUDA_VISIBLE_DEVICES=0 python3 probe_extractor.py
CUDA_VISIBLE_DEVICES=0 python3 probe_classifier.py > ./results/results.txt
python3 process_results.py
python3 plot_results.py