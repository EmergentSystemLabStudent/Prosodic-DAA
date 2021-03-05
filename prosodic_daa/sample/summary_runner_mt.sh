#!/bin/bash

label=sample_results

while getopts l: OPT
do
  case $OPT in
    "l" ) label="${OPTARG}" ;;
  esac
done

result_dirs=`ls ${label} | grep "[0-9]*"`


python3 summary_summary.py ${label} ${result_dirs}

cp -r summary_files ${label}
cp -r figures ${label}

rm -rf summary_files
rm -rf figures
rm -rf results
