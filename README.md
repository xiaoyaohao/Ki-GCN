# Ki-GCN
Ki-GCN (Kinship-Graph Convolutional Network), a novel kinship-based graph deep learning approach used to identify genome variants that are associated with diseases or traits.

The Ki-GCN is a graph autoencoder utilizing both the information from the genome variants matrix and the family structure information from the kinship matrix. It uses sample variants matrix and kinship matrix of samples as input, the output is the identified variants matirx.

The sample variants matrix is constructed from vcf files, every row represents a sample and every column represents a variant. The kinship matrix could be computed by IBS or other algorithms.

usage: python ki-gcn.py -v=variants_matrix.csv -k=kinship_matrix.csv -l=label.csv -e=edeg.csv -o output
Options and arguments:
-v: The variants matrix, each row represent a sample,and each column represent a variant.
-k: The kinship matrix calculated by ibs algorithm, it should keep one decimal place.
-l: The label of the samples, it should has the same order as variants matrix and the kinship matrix, the label should be binary.
-e: The edge matrix transformed from the kinship matrix, there are two columns, the first column is the order of samples, the second column is the neighbor of samples in the first column, the third column is the corresponding kinship coefficient.
-o: The prefix of the output file, which contains the selected variants matrix, the default is output_kigcn.
