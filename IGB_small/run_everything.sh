# echo "Running GCN 19 classes"
# python gnn.py --modelpath gcn_19.pt --num_classes 19 --model gcn > gcn_19.txt & timeout 10m nvidia-smi dmon -i 1 > gcn_19_gpu_stats.txt;
# wait
# echo "Running GAT 2983 classes"
python gnn.py --modelpath gat_2983.pt --num_classes 2983 --model gat > gat_2983.txt & timeout 10m nvidia-smi dmon -i 1 > gat_2983_gpu_stats.txt;
