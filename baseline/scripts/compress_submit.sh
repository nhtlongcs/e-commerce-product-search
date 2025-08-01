mkdir /home/mpham/workspace/source/analyticup/outputs/baseline/predictions/14B
cp /home/mpham/workspace/source/analyticup/outputs/baseline/predictions/submit_QC.txt /home/mpham/workspace/source/analyticup/outputs/baseline/predictions/14B/submit_QC.txt
cp /home/mpham/workspace/source/analyticup/outputs/baseline/predictions/submit_QI.txt /home/mpham/workspace/source/analyticup/outputs/baseline/predictions/14B/submit_QI.txt
cd /home/mpham/workspace/source/analyticup/outputs/baseline/predictions/14B
zip -FS ./submit.zip submit_QC.txt submit_QI.txt