#!/bin/bash

label=sample_results_aioi
begin=1
end=20

while getopts l:b:e: OPT
do
  case $OPT in
    "l" ) label="${OPTARG}" ;;
    "b" ) begin="${OPTARG}" ;;
    "e" ) end="${OPTARG}" ;;
  esac
done

sh runner_aioi.sh -l ${label}/npbdaa -b ${begin} -e ${end}

sh summary_runner_aioi.sh -l ${label}/npbdaa

sh clean.sh

sh runner_pdaa_aioi.sh -l ${label}/pdaa -b ${begin} -e ${end}

sh summary_runner_aioi.sh -l ${label}/pdaa

sh clean.sh

sh runner_pdaa_f0_aioi.sh -l ${label}/pdaa_f0 -b ${begin} -e ${end}

sh summary_runner_aioi.sh -l ${label}/pdaa_f0

sh clean.sh

sh runner_pdaa_sil_aioi.sh -l ${label}/pdaa_sil -b ${begin} -e ${end}

sh summary_runner_aioi.sh -l ${label}/pdaa_sil

sh clean.sh
