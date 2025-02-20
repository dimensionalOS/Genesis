#!/usr/bin/env bash
### every exit != 0 fails the script
set -e

echo "Install nss-wrapper to be able to execute image as non-root user"
apt-get update
apt-get install -y libnss-wrapper gettext
apt-get clean -y

echo "add 'source generate_container_user' to .bashrc"

# have to be added to hold all env vars correctly
echo 'source $STARTUPDIR/generate_container_user' >> $HOME/.bashrc
echo 'set -o vi' >> $HOME/.bashrc
echo 'source /opt/ros/humble/setup.bash' >> $HOME/.bashrc
echo 'source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash' >> $HOME/.bashrc
echo 'source /home/ros/dev_ws/install/setup.bash' >> $HOME/.bashrc