# Dropbox Path
DROPBOX_PATH="$HOME/Dropbox/Research/NeuralHamilton/"

# Join the two paths
FULL_PATH=$DROPBOX_PATH$1

# If the directory doesn't exist, create it
if [ ! -d $FULL_PATH ]; then
  mkdir $FULL_PATH
fi

# Link the directory to pwd
ln -s $FULL_PATH $PWD/$1

