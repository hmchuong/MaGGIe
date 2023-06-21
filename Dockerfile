FROM docker-matrix-experiments-snapshot.dr-uw2.adobeitc.com/runai/clio-base-demo:0.06

# Fix missing libGL.so.1
RUN apt install libgl1-mesa-glx -y

# Install pytorch
RUN pip install --no-cache-dir spconv-cu120 kornia==0.6.12 mmcv==1.5.0

# Entrypoint will chown the home folder to the uid
# ENTRYPOINT ["./run-ssh.sh"]
# CMD ["./execute.sh"]