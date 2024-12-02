import pandas as pd
from Bio import SeqIO
import csv
import os
import argparse
from CLEAN.infer import infer_maxsep
from CLEAN.utils import *
import shutil


os.environ['TORCH_HOME']='/sharefs/gongjj/data/torch_home'

train_data = "split100"


def main(args):
    
    # prepare data frame for CLEAN model
    ec_nums = ['1.1.1.37','2.7.1.71','3.1.1.2']

    for ec_num in ec_nums:

        sequences = []
        seq = []
        extracted_file = os.path.join(args.data_dir, f"ec_{ec_num}.a3m")
        for record in SeqIO.parse(extracted_file, "fasta"):
            sequences.append((str(record.id).split()[0], ec_num, str(record.seq).upper()))
            seq.append(str(record.seq).upper())

        with open(extracted_file.replace('a3m','csv'), 'w', newline='') as csvfile:
            # 创建 CSV writer 对象
            csv_writer = csv.writer(csvfile, delimiter='\t')
            
            # 写入表头
            csv_writer.writerow(['ID', 'EC', 'Sequences'])
            
            # 写入数据
            for row in sequences:
                csv_writer.writerow(row)

        print(f"for file {ec_num}, after dudip: {len(list(set(seq)))} uniqueness = {len(list(set(seq)))/len(seq)}")


    # CLEAN model inference
    subdir = args.data_dir.split('/')[-1]
    output_dir = os.path.join('./results', f"{subdir}_clean")
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)

    

    for file in os.listdir(args.data_dir):
        if(not file.endswith('.csv')):
            continue
        
        fpath = os.path.join(args.data_dir, file)
        filename = file[:-4]

        # copy files to backup
        output_fpath = os.path.join(output_dir, file)
        shutil.copy(fpath, output_fpath)

        test_data = f"{subdir}/{filename}"
        prepare_infer_fasta(test_data) 
        infer_maxsep(train_data, test_data, report_metrics=True, pretrained=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    args = parser.parse_args()
    main(args)