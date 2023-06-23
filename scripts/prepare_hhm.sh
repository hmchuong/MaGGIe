cd /mnt/localssd
aws s3 cp --recursive s3://chuongh-vm2m/HHM HHM
cd HHM

unzip -q alphas.zip
unzip -q HHM2K.zip
rm alphas.zip
rm HHM2K.zip

mv images train
mv HHM2K val

cd train
cat images.zip.* > images.zip
unzip images.zip
rm images.zip.*
rm images.zip