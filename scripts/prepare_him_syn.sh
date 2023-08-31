cd /mnt/localssd
if [ ! -d "HIM" ]; then
    rsync -av /sensei-fs/users/chuongh/data/vm2m/HIM2K.zip .
    echo "Unzipping HIM2K..."
    unzip -q HIM2K.zip
fi
# Prepare dataset
rsync -av /sensei-fs/users/chuongh/data/vm2m/BG20K/train.tar .
if [ ! -d "train" ]; then
    tar -xf train.tar
    mv train bg
fi
if [ ! -d "HHM" ]; then
    mkdir HHM
    cd HHM
    rsync -av /sensei-fs/users/chuongh/data/vm2m/hhm_synthesized.tar.gz .
    tar -xf hhm_synthesized.tar.gz
fi