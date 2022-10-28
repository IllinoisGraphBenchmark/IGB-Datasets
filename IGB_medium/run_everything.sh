echo "Running GCN 19 classes"
python gnn.py --modelpath gcn_19.pt --num_classes 19 --model gcn > gcn_19.txt & timeout 30m nvidia-smi dmon -i 2 > gcn_19_gpu_stats.txt;
wait
echo "Running GCN 2983 classes"
python gnn.py --modelpath gcn_2983.pt --num_classes 2983 --model gcn > gcn_2983.txt & timeout 30m nvidia-smi dmon -i 2 > gcn_2983_gpu_stats.txt;
wait
echo "Running SAGE 19 classes"
python gnn.py --modelpath gsage_19.pt --num_classes 19 --model sage > gsage_19.txt & timeout 30m nvidia-smi dmon -i 2 > gsage_19_gpu_stats.txt;
wait
echo "Running SAGE 2983 classes"
python gnn.py --modelpath gsage_2983.pt --num_classes 2983 --model sage > gsage_2983.txt & timeout 30m nvidia-smi dmon -i 2 > gsage_2983_gpu_stats.txt;
wait
echo "Running GAT 19 classes"
python gnn.py --modelpath gat_19.pt --num_classes 19 --model gat > gat_19.txt & timeout 30m nvidia-smi dmon -i 2 > gat_19_gpu_stats.txt;
wait
echo "Running GAT 2983 classes"
python gnn.py --modelpath gat_2983.pt --num_classes 2983 --model gat > gat_2983.txt & timeout 30m nvidia-smi dmon -i 2 > gat_2983_gpu_stats.txt;
