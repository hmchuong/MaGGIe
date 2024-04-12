cd /mnt/localssd
if [ ! -d "HIM2K" ]; then
    rsync -av /sensei-fs/users/chuongh/data/vm2m/HIM2K.zip .
    echo "Unzipping HIM2K..."
    unzip -q HIM2K.zip
    cd HIM2K
    rsync -av  /sensei-fs/users/chuongh/data/vm2m/masks_matched_r50_fpn_3x.zip .
    unzip -q masks_matched_r50_fpn_3x.zip
    
    # Copy combine set
    cd images 
    rsync -av /sensei-fs/users/chuongh/data/vm2m/combine_images.zip .
    unzip -q combine_images.zip
    cd ..
    
    cd alphas 
    rsync -av /sensei-fs/users/chuongh/data/vm2m/combine_alphas.zip .
    unzip -q combine_alphas.zip
    cd ..

    cd masks
    rsync -av /sensei-fs/users/chuongh/data/vm2m/combine_masks.zip .
    unzip -q combine_masks.zip
    cd ..
    
    cd ..
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