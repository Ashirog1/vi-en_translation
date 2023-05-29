#!/bin/bash

# Set the path to the Moses scripts directory
SCRIPTS=$PWD/mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl

# Set the path to the reference and hypothesis files
file_ref=$PWD/reference.txt
file_hyp=$PWD/hypothesis.txt

# Tokenize the reference and hypothesis files using the Moses tokenizer
cat $file_ref | $TOKENIZER -l vi > file.ref
cat $file_hyp | $TOKENIZER -l vi > file.hyp

# Calculate BLEU score using sacreBLEU
sacrebleu -tok '13a' -s 'exp' file.ref < file.hyp