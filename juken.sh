#!/bin/sh

echo "5"
python eiken.py -f grade_5    > ./result/grade_5

echo "4"
python eiken.py -f grade_4    > ./result/grade_4

echo "3"
python eiken.py -f grade_3    > ./result/grade_3

echo "pre2"
python eiken.py -f grade_pre2 > ./result/grade_pre2

echo "2"
python eiken.py -f grade_2    > ./result/grade_2

echo "pre1"
python eiken.py -f grade_pre1 > ./result/grade_pre1

echo "1"
python eiken.py -f grade_1    > ./result/grade_1