import os
import glob
import argparse
import re

parser = argparse.ArgumentParser(description="Fix the duration property in the movie files")
parser.add_argument("input_folder")
parser.add_argument("output_folder")
args = parser.parse_args()

print(args.input_folder, '>>>', args.output_folder)

os.makedirs(args.output_folder, exist_ok=True)

movies = sorted(glob.glob(os.path.join(args.input_folder, '*.mp4')))

for movie in movies:
  bn = os.path.basename(movie)
  cmd = " ".join(["ffmpeg",
                  "-ignore_editlist 1",
                  "-i " + re.escape(movie),
                  "-codec copy",
                  os.path.join(re.escape(args.output_folder), re.escape(bn))])
  print(cmd)
  os.system(cmd)

print('done.')
