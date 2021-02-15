passwd="1q2w3e4r"

if [ "$(uname)" == "Darwin" ]; then
    home_dir=$HOME
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # presumably on Colab
    home_dir="/content"
    # sshpass is required to connect ssh in Colab
    sudo apt install -qq sshpass
else
    echo "Windows not supprted"; exit
fi

login_ssh(){
    echo "logging into server..."

    if [ "$(uname)" == "Darwin" ]; then
        ssh -o StrictHostKeyChecking=no server@isys313.iptime.org -p 80
    else
        sshpass -p $passwd \
        ssh -o StrictHostKeyChecking=no server@isys313.iptime.org -p 80
    fi
}

mount_sshfs() {
    # install based on platform
    if [ "$(uname)" == "Darwin" ]; then
        brew install sshfs
    else
        sudo apt install -qq sshfs
    fi

    killall sshfs
    if [ -d "$home_dir/data_server" ]; then
        sudo umount -f data_server
        rmdir "$home_dir/data_server"
    fi
    mkdir "$home_dir/data_server"

    sudo sshfs \
        -o password_stdin,allow_other,reconnect,StrictHostKeyChecking=no \
        server@isys313.iptime.org:/home/server/duhyeuk "$home_dir/data_server" \
        -p 80 <<< $passwd
    echo '\n'
    echo "mount successful. To unmount, type:"
    echo "$ killall sshfs; sudo umount -f $home_dir/data_server"
}

# main screen
echo "┌━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┐ "
echo "  Welcome to OpenAI -  Eagle Eyes Member " $USER "님"
echo "└━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┘ "
echo "  \n"
echo "  server에 접근하시겠습니까??"
echo "  (password : 1q2w3e4r)"
echo "  options:"

select opt in "1. login to server" "2. mount to local device"; do
    case $opt in
        "1. login to server" ) login_ssh; break;;
        "2. mount to local device" ) mount_sshfs; exit;;
    esac
done

