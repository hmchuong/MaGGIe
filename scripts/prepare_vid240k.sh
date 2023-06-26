
DATASET=/mnt/localssd/VideoMatte240K
cd /mnt/localssd/

if [ ! -d "$DATASET" ]; then
    # Copy original dataset
    # rsync -av /sensei-fs/users/chuongh/VideoMatte240K.tar.gz .
    echo 'Copying main data...'
    rsync -av /sensei-fs/tenants/Sensei-AdobeResearchTeam/videomasking/human_matting_data/VideoMatte240K.tar .
    tar -xf VideoMatte240K.tar

    echo 'Copying masks for testing...'
    # Copy composite images for testing
    cd VideoMatte240K/test
    # copy and extract composite images
    rsync -av /sensei-fs/users/chuongh/data/vm2m/VideoMatte240K/test/comp.tar .
    tar -xf comp.tar

    # copy and extract random binarized masks 
    rsync -av /sensei-fs/users/chuongh/data/vm2m/VideoMatte240K/test/coarse.tar .
    tar -xf coarse.tar

    # Copy testing masks propagation

    # Copy testing trimaps from alphas

    # Copy testing trimaps propagation

    echo 'Copying masks for validation...'
    # Copy composite images for validation
    cd ../valid
    # copy and extract composite images
    rsync -av /sensei-fs/users/chuongh/data/vm2m/VideoMatte240K/valid/comp.tar .
    tar -xf comp.tar

    # copy and extract random binarized masks 
    rsync -av /sensei-fs/users/chuongh/data/vm2m/VideoMatte240K/valid/coarse.tar .
    tar -xf coarse.tar

    # Copy background images for training
    echo "Copying background images for training..."
    cd /mnt/localssd/
    rsync -av /sensei-fs/users/chuongh/data/vm2m/BG20K/train.tar .
    tar -xf train.tar
    mv train bg
fi