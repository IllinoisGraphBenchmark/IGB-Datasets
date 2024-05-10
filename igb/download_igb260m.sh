echo("IGB-full (Homogeneous) download starting");
# paper
wget --recursive --no-parent https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/node_feat.npy
wget --recursive --no-parent https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/node_label_19.npy
wget --recursive --no-parent https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/node_label_2K.npy
wget --recursive --no-parent https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper/paper_id_index_mapping.npy

# paper__cites__paper
wget --recursive --no-parent https://igb-public-awsopen.s3.amazonaws.com/IGBH/processed/paper__cites__paper/edge_index.npy
echo("IGB-full (Homogeneous) download complete");

