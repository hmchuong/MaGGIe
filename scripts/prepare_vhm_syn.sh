cd /mnt/localssd
# Install s5cmd

# sudo apt-get install -y unzip zip
if [ ! -d "syn" ]; then
    # aws s3 cp --recursive s3://chuongh-vm2m/vhm-syn vhm-syn
    # cd vhm-syn
    # zip -F syn.zip --out ../single-syn.zip
    # cd ..
    # unzip -q single-syn.zip
    wget https://github.com/peak/s5cmd/releases/download/v2.2.1/s5cmd_2.2.1_Linux-64bit.tar.gz -O s5cmd.tar.gz
    tar -xf s5cmd.tar.gz
    chmod +x s5cmd
    sudo mv s5cmd /usr/local/bin/s5cmd
    s5cmd cp -n -s -u --sp "s3://a-chuonghm/syn_new/*" syn/
fi
 