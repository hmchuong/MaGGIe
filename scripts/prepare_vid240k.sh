
DATASET=/mnt/localssd/VideoMatte240K
cd /mnt/localssd/

if [ ! -d "$DATASET" ]; then
    # Copy original dataset
    rsync -av /sensei-fs/users/chuongh/VideoMatte240K.tar.gz .
    tar -xf VideoMatte240K.tar.gz

    # Copy composite images

    # Copy testing masks

    # Copy background images
    rsync -av /sensei-fs/users/chuongh/bg.tar.gz .
    tar -xf bg.tar.gz
fi