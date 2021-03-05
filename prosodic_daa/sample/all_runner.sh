#!/bin/bash

label=sample_results
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

sh runner_aioi.sh -l ${label}/aioi/npbdaa -b ${begin} -e ${end}

sh summary_runner_aioi.sh -l ${label}/aioi/npbdaa

sh clean.sh

sh runner_pdaa_aioi.sh -l ${label}/aioi/pdaa -b ${begin} -e ${end}

sh summary_runner_aioi.sh -l ${label}/aioi/pdaa

sh clean.sh

sh runner_pdaa_f0_aioi.sh -l ${label}/aioi/pdaa_f0 -b ${begin} -e ${end}

sh summary_runner_aioi.sh -l ${label}/aioi/pdaa_f0

sh clean.sh

sh runner_pdaa_sil_aioi.sh -l ${label}/aioi/pdaa_sil -b ${begin} -e ${end}

sh summary_runner_aioi.sh -l ${label}/aioi/pdaa_sil

sh clean.sh

sh runner_murakami.sh -l ${label}/murakami/npbdaa -b ${begin} -e ${end}

sh summary_runner_murakami.sh -l ${label}/murakami/npbdaa

sh clean.sh

sh runner_pdaa_murakami.sh -l ${label}/murakami/pdaa -b ${begin} -e ${end}

sh summary_runner_murakami.sh -l ${label}/murakami/pdaa

sh clean.sh

sh runner_pdaa_f0_murakami.sh -l ${label}/murakami/pdaa_f0 -b ${begin} -e ${end}

sh summary_runner_murakami.sh -l ${label}/murakami/pdaa_f0

sh clean.sh

sh runner_pdaa_sil_murakami.sh -l ${label}/murakami/pdaa_sil -b ${begin} -e ${end}

sh summary_runner_murakami.sh -l ${label}/murakami/pdaa_sil

sh clean.sh

sh runner_power_law.sh -l ${label}/power_law/npbdaa -b ${begin} -e ${end}

sh summary_runner_power_law.sh -l ${label}/power_law/npbdaa

sh clean.sh

sh runner_pdaa_power_law.sh -l ${label}/power_law/pdaa -b ${begin} -e ${end}

sh summary_runner_power_law.sh -l ${label}/power_law/pdaa

sh clean.sh

sh runner_pdaa_f0_power_law.sh -l ${label}/power_law/pdaa_f0 -b ${begin} -e ${end}

sh summary_runner_power_law.sh -l ${label}/power_law/pdaa_f0

sh clean.sh

sh runner_pdaa_sil_power_law.sh -l ${label}/power_law/pdaa_sil -b ${begin} -e ${end}

sh summary_runner_power_law.sh -l ${label}/power_law/pdaa_sil

sh clean.sh
