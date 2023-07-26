cd /mnt/localssd
if [ ! -d "HHM" ]; then
    aws s3 cp --recursive s3://chuongh-vm2m/HHM HHM
    cd HHM
    echo "Unzipping alphas and HHM2K..."
    unzip -q alphas.zip
    unzip -q HHM2K.zip
    rm alphas.zip
    rm HHM2K.zip

    mv images train
    mv HHM2K val
    mv alphas train/alphas
    echo "Unzipping images..."
    cd train
    cat images.zip.* > images.zip
    unzip -q images.zip
    rm images.zip.*
    rm images.zip
fi

echo "Cleaning HHM..."
cd /sensei-fs/users/chuongh/vm2m/tools
python clean_hhm.py