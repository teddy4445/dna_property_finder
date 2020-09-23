#!/usr/bin/python

### Created by Lior Zimmerman (http://www.github.com/LiorZ) ###
### Distributed under MIT License (http://opensource.org/licenses/MIT) ###

import sys, getopt
from Bio import SeqIO,pairwise2
import Bio.SubsMat.MatrixInfo as matrices
import sklearn.cluster as cluster

def clustering(seq_file,num_clusters,matrix,gap_open,gap_extend):

    handle = open(seq_file, "rU")

    records = list(SeqIO.parse(handle, "fasta"))
    num_seqs = len(records)

    scores = [[0 for i in range(num_seqs)] for j in range(num_seqs)]

    for i in range(0,num_seqs):
        for j in range(0,num_seqs):
            a = pairwise2.align.globalds(records[i],records[j],matrix,gap_open,gap_extend)
            (s1,s2,score,start,end) = a[0]
            scores[i][j] = score

    kmeans = cluster.KMeans(num_clusters)
    results = kmeans.fit(scores)

    labels = results.labels_
    clusters = [[] for i in range(num_clusters)]

    for i in range(0,num_seqs):
        clusters[labels[i]].append(records[i])

    for i in range(0,len(clusters)):
        output_handle = open("c."+str(i)+".fasta", "w")
        SeqIO.write(clusters[i], output_handle, "fasta")
        output_handle.close()


def main(argv):
    inputfile = None
    outputfile = ''
    num_clusters = 2
    matrix = "blosum62"
    gap_open = -10
    gap_extend = -0.5
    err_format = """
KMeans clustering script.
This file will cluster a list of sequences based on a substitution matrix and will output each cluster in a separate file.
Available flags:
-i (--input)            Input sequences file (*required*)
-n (--num_clusters)     Number of clusters (default=2)
-m (--matrix)           Substitution matrix (default=blosum62)
-o (--gap_open)         Gap open penalty
-e (--gap_extend)       Gap Extend penalty
    """
    try:
        opts, args = getopt.getopt(argv,"hi:n:m:o:e:",["input=","num_clusters=","matrix=","gap_open=","gap_extend"])
    except getopt.GetoptError:
        print err_format
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print err_format
            sys.exit()
        elif opt in ("-i", "--input"):
            inputfile = arg
        elif opt in ("-n", "--num_clusters"):
            num_clusters = arg
        elif opt in ("-m","--matrix"):
            matrix = arg

    if inputfile == None:
        print "ERROR: Input Sequences file is missing"
        print err_format
        sys.exit()

    print "Starting clustering with matrix " + matrix
    clustering(inputfile,num_clusters,getattr(matrices,matrix),gap_open,gap_extend)


if __name__ == "__main__":
   main(sys.argv[1:])