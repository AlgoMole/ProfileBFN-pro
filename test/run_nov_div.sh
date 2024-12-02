#!/bin/bash

data_dir="$1"

python test/calc_div.py --input_dir ${data_dir}

python test/calc_nov.py --input_dir ${data_dir}