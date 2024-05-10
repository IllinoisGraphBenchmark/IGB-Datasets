echo("IGB-large (Homogeneous) download starting");
# paper
wget https://igb-public-awsopen.s3.amazonaws.com/igb_large/processed/paper/node_feat.npy
wget https://igb-public-awsopen.s3.amazonaws.com/igb_large/processed/paper/node_label_19.npy
wget https://igb-public-awsopen.s3.amazonaws.com/igb_large/processed/paper/node_label_2K.npy
wget https://igb-public-awsopen.s3.amazonaws.com/igb_large/processed/paper/paper_id_index_mapping.npy

# paper__cites__paper
wget --recursive --no-parent https://igb-public-awsopen.s3.amazonaws.com/igb_large/processed/paper__cites__paper/edge_index.npy
echo("IGB-large (Homogeneous) download complete");
