FROM docker-jorb-release.dr-uw2.adobeitc.com/cliodev_cuda12:0.1.22


# needed for ubuntu auto install. Otherwise it prompts for tzdata 
ENV DEBIAN_FRONTEND=noninteractive

###
# Common installs
###
RUN apt-get update && \
    apt-get -y install apt-utils sudo wget curl openssh-server dnsutils git make build-essential ca-certificates rsyslog sssd && \
    pip install jupyterlab tensorflow tensorboard

# Make /home/user default for installations
WORKDIR /home/user
ENV HOME /home/user
ENV PATH="/home/user:/home/user/.local/bin:$PATH"

# Give all users sudo permissions, this is required for following reasons:
# - apt-get installs from container
# - to be able to chown the /home/user folder so that you dont see PermissionDenied errors 
RUN echo "%users ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/override

###
# SSH section
###

COPY entrypoint.sh run.sh /
COPY start-notebook.sh /home/user
RUN mkdir -p /etc/sssd/conf.d && chmod 711 /etc/sssd
COPY sssd.conf /etc/sssd
RUN chmod 0600 /etc/sssd/sssd.conf
EXPOSE 8888 6006 22
RUN chmod 777 /entrypoint.sh /run.sh /home/user/start-notebook.sh

# populate sshd server options
RUN echo "\n\
PasswordAuthentication yes \n\
PermitRootLogin no \n\
MaxAuthTries 3 \n\
LoginGraceTime 10 \n\
PermitEmptyPasswords no \n\
ChallengeResponseAuthentication no \n\
KerberosAuthentication no \n\
GSSAPIAuthentication no \n\
X11Forwarding no \n\
AuthorizedKeysCommand /usr/bin/sss_ssh_authorizedkeys \n\
AuthorizedKeysCommandUser nobody \n\
AuthenticationMethods publickey \n\
PubkeyAuthentication yes \n\
UsePAM yes \n\
SyslogFacility AUTH \n\
LogLevel VERBOSE \n\
" >> /etc/ssh/sshd_config

# Entrypoint will chown the home folder to the uid
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/run.sh"]