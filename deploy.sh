#!/bin/sh
# Copy the phone-side files over WiFi ssh. Nothing here touches the gadget:
# deploying is safe, `gadget.sh add` is the step that unbinds USB.
#
#   ./deploy.sh user@192.168.0.18            # tenta /opt/j5, usa sudo se precisar
#   DEST=j5 ./deploy.sh user@192.168.0.18    # ~/j5, sem sudo nenhum para copiar
#
# Non-root targets: /opt/j5 usually belongs to root, so the script falls back to
# sudo for the mkdir and hands the directory to the login user. With DEST=j5 the
# copy needs no sudo at all, only running gadget.sh/hidmouse.py does.

set -eu

TARGET="${1:-${TARGET:-root@samsung-j5.local}}"
DEST="${DEST:-/opt/j5}"
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
FILES="protocol.py report_desc.py hidmouse.py gadget.sh j5-hid.initd"

# Preflight: a half copied checkout is the most common way this fails, and it
# fails later and more confusingly if we let scp discover it.
missing=""
for name in $FILES; do
    [ -f "$HERE/$name" ] || missing="$missing $name"
done
if [ -n "$missing" ]; then
    echo "erro: faltam arquivos em $HERE:$missing" >&2
    echo "      rode este script do diretorio que tem o projeto completo" >&2
    exit 1
fi

echo "destino: $TARGET:$DEST"

# "consegui criar" is not the same as "posso escrever": /opt/j5 may already exist
# owned by root, in which case mkdir -p succeeds and scp fails afterwards.
if ! ssh "$TARGET" "mkdir -p '$DEST' 2>/dev/null && test -w '$DEST'" 2>/dev/null; then
    echo "$DEST nao e gravavel por este usuario, tentando com sudo"
    if ! ssh -t "$TARGET" "sudo mkdir -p '$DEST' && sudo chown \$(id -u):\$(id -g) '$DEST'"; then
        echo "erro: nao consegui preparar $DEST." >&2
        echo "      sem sudo no aparelho? use: DEST=j5 $0 $TARGET" >&2
        exit 1
    fi
fi

scp "$HERE/protocol.py" "$HERE/report_desc.py" "$HERE/hidmouse.py" "$HERE/gadget.sh" \
    "$HERE/j5-hid.initd" "$TARGET:$DEST/"
ssh "$TARGET" "chmod +x '$DEST/gadget.sh' '$DEST/hidmouse.py' '$DEST/j5-hid.initd'"

echo
echo "verificando o que chegou:"
ssh "$TARGET" "ls -l '$DEST'"

# Relative DEST is relative to the login directory; the hints must be absolute or
# they break the moment someone runs them from a different cwd.
case "$DEST" in
    /*) REMOTE_DIR="$DEST" ;;
    *)
        REMOTE_HOME=$(ssh "$TARGET" 'pwd')
        REMOTE_DIR="$REMOTE_HOME/$DEST"
        ;;
esac

echo
echo "o gadget e o /dev/hidg0 sao de root, entao no aparelho:"
echo "  cd $REMOTE_DIR"
echo "  sudo ./gadget.sh status"
echo "  sudo ./gadget.sh add"
echo "  sudo nohup ./hidmouse.py --stats >/tmp/hid.log 2>&1 &"

if [ "$REMOTE_DIR" != "/opt/j5" ]; then
    echo
    echo "fora de /opt/j5: se for instalar o servico de boot, crie /etc/conf.d/j5-hid com"
    echo "  J5_DIR=$REMOTE_DIR"
fi