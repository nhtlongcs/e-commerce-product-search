mkdir 14B
cp predictions/submit_QC.txt predictions/14B/submit_QC.txt
cp predictions/submit_QI.txt predictions/14B/submit_QI.txt
cd predictions/14B
zip -FS ./submit.zip submit_QC.txt submit_QI.txt