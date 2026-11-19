#!/bin/sh
# j5-gadget: adds a HID mouse function to the postmarketOS USB gadget.
#
# The pmOS initramfs builds a gadget called g1 (CDC-NCM only) on every boot and
# leaves it bound to the UDC, so the network interface you may be using survives
# switch_root. We do not fight that: this script adds the HID function into the
# existing g1, rebinds, and `del` puts everything back exactly as it was.
#
# Usage:
#   gadget.sh add            add the absolute mouse (5-byte report)
#   gadget.sh add --rel      add a relative mouse instead (3-byte report)
#   gadget.sh add --with-acm also expose a CDC-ACM serial (ttyGS0 on the phone)
#   gadget.sh add --with-rndis also expose RNDIS (Windows 10 networking)
#   gadget.sh del            remove what add created, keep g1 as the initramfs left it
#   gadget.sh status         show what is bound right now
#
# NOTE: add/del unbind the gadget for ~1s, which drops usb0 on the host.
#       If your shell is on usb0 you will lock yourself out. Use WiFi ssh.

set -eu

GADGET="${GADGET:-g1}"
CFS="${CFS:-/sys/kernel/config/usb_gadget}"
FUNC="${FUNC:-hid.usb0}"
# Overridable so the script can be exercised against a fake configfs tree
# (selftest.py does exactly that) and so a different hidg node can be used.
UDC_OVERRIDE="${UDC:-}"
HIDG="${HIDG:-/dev/hidg0}"
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
G="$CFS/$GADGET"

log() { printf '%s\n' "$*" >&2; }
die() { log "erro: $*"; exit 1; }

# configfs attributes accept open() happily and then fail the write with EINVAL,
# which printf reports only through its exit status -- under `set -e` that aborts
# with no message at all. Always write through here, and read back.
write_attr() {
    if ! printf '%s' "$2" > "$1" 2>/dev/null; then
        die "escrita falhou: '$2' -> $1"
    fi
    got=$(cat "$1" 2>/dev/null || true)
    [ "$got" = "$2" ] || die "valor nao pegou em $1: escrevi '$2', li '$got'"
}

# If we die between unbind and bind, the gadget stays down and usb0 is gone
# until the next boot. Put it back on the way out, whatever happened.
restore_on_exit() {
    status=$?
    if [ "$status" -ne 0 ] && [ -n "${UDC_WAS:-}" ] && [ -e "$G/UDC" ]; then
        log "restaurando $GADGET em $UDC_WAS por causa da falha"
        printf '%s' "$UDC_WAS" > "$G/UDC" 2>/dev/null || log "aviso: nao consegui reatar"
    fi
}

find_udc() {
    if [ -n "$UDC_OVERRIDE" ]; then
        printf '%s' "$UDC_OVERRIDE"
        return 0
    fi
    udc=$(ls /sys/class/udc 2>/dev/null | head -n 1 || true)
    [ -n "$udc" ] || die "nenhum UDC em /sys/class/udc (o kernel nao tem controlador gadget?)"
    printf '%s' "$udc"
}

find_cfg() {
    cfg=$(ls -d "$G"/configs/*/ 2>/dev/null | head -n 1 || true)
    [ -n "$cfg" ] || die "nenhuma configuracao em $G/configs"
    printf '%s' "${cfg%/}"
}

find_cfg_quiet() {
    cfg=$(ls -d "$G"/configs/*/ 2>/dev/null | head -n 1 || true)
    printf '%s' "${cfg%/}"
}

# Alpine's busybox has fancy sleep, but a fallback keeps the wait loop honest
# instead of collapsing into 50 back-to-back iterations that always fail.
pause() {
    sleep "$1" 2>/dev/null || sleep 1
}

unbind() {
    bound=$(cat "$G/UDC" 2>/dev/null || true)
    if [ -n "$bound" ]; then
        UDC_WAS="$bound"
        log "desatando $GADGET de $bound"
        echo "" > "$G/UDC"
        pause 0.4
    fi
}

bind() {
    udc=$(find_udc)
    log "atando $GADGET em $udc"
    printf '%s' "$udc" > "$G/UDC"
    i=0
    while [ "$i" -lt 50 ]; do
        if [ -e "$HIDG" ]; then
            log "ok: $HIDG presente"
            return 0
        fi
        pause 0.1
        i=$((i + 1))
    done
    die "$HIDG nao apareceu. dmesg | tail -30: $(dmesg | tail -3 | tr '\n' ' ')"
}

add_hid() {
    report_len=$1
    rel_flag=$2
    cfg=$(find_cfg)

    # f_hid refuses to replace a descriptor that is already set (-EBUSY), so a
    # mode switch has to recreate the function instead of reconfiguring it. The
    # gadget is already unbound here, which is what makes the removal safe.
    if [ -L "$cfg/$FUNC" ]; then
        rm -f "$cfg/$FUNC"
    fi
    if [ -d "$G/functions/$FUNC" ]; then
        log "recriando a funcao $FUNC"
        rmdir "$G/functions/$FUNC" 2>/dev/null || rm -rf "$G/functions/$FUNC"
    fi

    mkdir -p "$G/functions/$FUNC" || die "nao consegui criar $G/functions/$FUNC"

    # The device advertises a full report descriptor, so it must NOT claim to
    # implement the boot protocol: boot mice are relative-only and the host would
    # ignore our absolute axes.
    write_attr "$G/functions/$FUNC/report_length" "$report_len"
    write_attr "$G/functions/$FUNC/subclass" 0
    write_attr "$G/functions/$FUNC/protocol" 0

    if [ "$rel_flag" = "1" ]; then
        python3 "$HERE/report_desc.py" "$G/functions/$FUNC/report_desc" --rel || die "report_desc.py --rel falhou"
    else
        python3 "$HERE/report_desc.py" "$G/functions/$FUNC/report_desc" || die "report_desc.py falhou"
    fi

    ln -sfn "$G/functions/$FUNC" "$cfg/$FUNC" || die "nao consegui ligar a funcao na config"
    log "ligado em $cfg/$FUNC"
}

add_net() {
    kind=$1   # acm or rndis
    name="$kind.usb0"
    mkdir -p "$G/functions/$name"
    cfg=$(find_cfg)
    ln -sfn "$G/functions/$name" "$cfg/$name"
    log "ligado em $cfg/$name"
}

status() {
    printf 'gadget      %s\n' "$G"
    if [ ! -d "$G" ]; then
        printf '%s\n' "  (nao existe)"
        return 0
    fi
    printf 'udc         %s\n' "$(cat "$G/UDC" 2>/dev/null)"
    printf 'idVendor    %s\n' "$(cat "$G/idVendor" 2>/dev/null)"
    printf 'idProduct   %s\n' "$(cat "$G/idProduct" 2>/dev/null)"
    printf 'functions   %s\n' "$(ls "$G/functions" 2>/dev/null | tr '\n' ' ')"
    cfg=$(find_cfg_quiet)
    if [ -n "$cfg" ]; then
        printf 'config %s  %s\n' "$cfg" "$(ls "$cfg" 2>/dev/null | tr '\n' ' ')"
    else
        printf 'config      (nenhuma)\n'
    fi
    printf 'state       %s\n' "$(cat /sys/class/udc/*/state 2>/dev/null | tr '\n' ' ')"
    if [ -e "$HIDG" ]; then
        printf 'hidg        %s presente\n' "$HIDG"
    else
        printf 'hidg        ausente (%s)\n' "$HIDG"
    fi
    if [ -e /dev/ttyGS0 ]; then
        printf 'ttyGS       /dev/ttyGS0 presente\n'
    fi
    printf 'usb0        %s\n' "$(ip addr show usb0 2>/dev/null | grep 'inet ' | awk '{print $2}' | head -n1 || true)"
}

del_hid() {
    cfg=$(find_cfg)
    if [ -L "$cfg/$FUNC" ]; then
        rm -f "$cfg/$FUNC"
        log "removido $cfg/$FUNC"
    fi
    if [ -d "$G/functions/$FUNC" ]; then
        rmdir "$G/functions/$FUNC" 2>/dev/null || log "aviso: $G/functions/$FUNC nao ficou vazio"
    fi
}

cmd="${1:-status}"
shift 2>/dev/null || true

REL=0
WITH_ACM=0
WITH_RNDIS=0
for arg in "$@"; do
    case "$arg" in
        --rel) REL=1 ;;
        --with-acm) WITH_ACM=1 ;;
        --with-rndis) WITH_RNDIS=1 ;;
        *) die "argumento desconhecido: $arg" ;;
    esac
done

[ -d "$G" ] || die "gadget $G nao existe. O initramfs do pmOS cria o g1 no boot; se nao existe, veja o README."

UDC_WAS=""
trap restore_on_exit EXIT
trap 'exit 130' INT TERM

case "$cmd" in
add)
    unbind
    # Subclass/protocol 0 means "use the report descriptor", which is the only
    # way to get the absolute axes through. Order matters: the function has to
    # exist and be linked into the config before the UDC is bound.
    if [ "$REL" = "1" ]; then
        add_hid 3 1
    else
        add_hid 5 0
    fi
    if [ "$WITH_ACM" = "1" ]; then add_net acm; fi
    if [ "$WITH_RNDIS" = "1" ]; then add_net rndis; fi
    bind
    status
    ;;
del)
    unbind
    del_hid
    if [ -d "$G/functions/acm.usb0" ]; then
        rm -f "$(find_cfg)/acm.usb0"
        rmdir "$G/functions/acm.usb0" 2>/dev/null || true
    fi
    if [ -d "$G/functions/rndis.usb0" ]; then
        rm -f "$(find_cfg)/rndis.usb0"
        rmdir "$G/functions/rndis.usb0" 2>/dev/null || true
    fi
    bind
    status
    ;;
status)
    status
    ;;
*)
    die "uso: $0 add|del|status [--rel] [--with-acm] [--with-rndis]"
    ;;
esac