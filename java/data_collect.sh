#!/bin/bash

LANGUAGES=("french" "hindi" "chinese")
NUM_SENTENCES=(10000 20000 30000 40000 50000)


for lang in "${LANGUAGES[@]}";
do
  for num in "${NUM_SENTENCES[@]}";
  do
    filename="result_model2/"$lang$num".out"
    java -cp ~/cs224n/pa1/java/classes cs224n.assignments.WordAlignmentTester -dataPath /afs/ir/class/cs224n/pa1/data -model cs224n.wordaligner.Model2 -language $lang -evalSet dev -trainSentences $num > $filename 2>&1
  done

done


