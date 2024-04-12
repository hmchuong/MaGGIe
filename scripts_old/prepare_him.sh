cd /mnt/localssd
if [ ! -d "HIM" ]; then
    rsync -av /sensei-fs/users/chuongh/data/vm2m/HIM2K.zip .
    echo "Unzipping HIM2K..."
    unzip -q HIM2K.zip
fi