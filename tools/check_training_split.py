"""Check an optional NEW training run against the original frozen memberships."""
from pathlib import Path
import argparse,csv
ROOT=Path(__file__).resolve().parents[1]
def rows(p):
 with p.open(encoding='utf-8-sig',newline='')as f:return list(csv.DictReader(f))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('new_split',type=Path);args=parser.parse_args()
 old=rows(ROOT/'Reproduction/Task_A/evidence/split.csv');new=rows(args.new_split)
 old={(r['id'],r['partition'],r['sha256'])for r in old};new={(r['id'],r['partition'],r['sha256'])for r in new}
 if new!=old:raise SystemExit('FAIL: split differs. Do not substitute this run for the frozen experiment.')
 print('PASS: all original train/development memberships and hashes match.')
