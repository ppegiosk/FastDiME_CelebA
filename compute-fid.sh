TEMPPATH=/scratch/ppar/tmp
OUTPATH=/scratch/ppar/results
EXPPATH=fastdime

mkdir -p ${TEMPPATH}/real
mkdir -p ${TEMPPATH}/cf
mkdir -p ${TEMPPATH}/cfmin

echo 'Copying CF images '

cp -r ${OUTPATH}/Results/${EXPPATH}/CC/CCF/CF/* ${TEMPPATH}/cf
cp -r ${OUTPATH}/Results/${EXPPATH}/IC/CCF/CF/* ${TEMPPATH}/cf

echo 'Copying real images'

cp -r ${OUTPATH}/Original/Correct/* ${TEMPPATH}/real
cp -r ${OUTPATH}/Original/Incorrect/* ${TEMPPATH}/real

echo 'Computing FID'

python -m pytorch_fid ${TEMPPATH}/real ${TEMPPATH}/cf --device cuda:0

rm -rf ${TEMPPATH}
