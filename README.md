# j5 — o Galaxy J5 como mouse USB real, movido por visão

O J5 (2015, postmarketOS, kernel mainline `msm8916`) vira um **dispositivo HID
físico**: o PC vê um mouse de verdade no barramento, sem driver e sem software de
injeção. Quem decide para onde apontar é a visão, que roda no PC (captura de tela
ou a câmera de outro celular).

```
[câmera ou tela] -> detecção -> (x,y) -> USB -> J5 -> /dev/hidg0 -> cursor
```

## O que o hardware permite (verificado, não suposto)

| Item | Estado |
|---|---|
| UDC | `ci_hdrc.0` (controlador ChipIdea) |
| `CONFIG_USB_CONFIGFS_F_HID` | `=y` no kernel do pmOS, sem rebuild |
| gadget do boot | `g1`, criado pelo initramfs, com `ncm.usb0` |
| câmera do J5 | **não existe** no mainline: o DTS não tem nó de câmera e os sensores (S5K3L2/S5K5E3) não têm driver |

A câmera do J5 está fora. Por isso a visão roda no PC e o aparelho é o atuador —
que é, aliás, o papel em que ele é insubstituível.

## Atualizar

O projeto vive em https://github.com/MI3SZ/autoaim.

```sh
# primeira vez (o diretório antigo não era repositório)
cd ~/Documents && mv j5 j5-antigo
git clone https://github.com/MI3SZ/autoaim.git j5

# depois de qualquer mudança
cd ~/Documents/j5 && git pull --ff-only
```

Se algum arquivo do celular mudar, o pull não basta — o aparelho roda a cópia
dele:

```sh
cd ~/Documents/j5 && DEST=j5 ./deploy.sh user@192.168.0.18
```

## Verificação

```sh
python3 selftest.py --bench
```

43 checagens e nenhuma precisa do aparelho, de câmera ou de display: protocolo,
descriptor HID, detector (banda de rolling shutter, moiré + JPEG q60, falso
positivo por cor, adaptação a blob implausível), calibração (homografia, polinômio
com distorção de lente, afinidade, save/load), pacing do replay, atuador
(deadband, reaquisição, teto de passo, watchdog, reinício do sender, modo
relativo), o `gadget.sh` inteiro contra uma árvore configfs falsa — incluindo o
caminho de falha que precisa reatar o UDC — e um laço completo detector → UDP →
atuador.

Custo por frame do detector, medido nesta máquina (opencv headless, `--bench`):

| fonte | por frame | teto teórico |
|---|---|---|
| 720p | 1,1 ms | ~890 fps |
| 1080p | 1,7 ms | ~575 fps |

A primeira versão levava 5,7 ms em 1080p: o `_mask` montava quatro temporários
float32 de frame inteiro por iteração. Hoje é um único `cv2.inRange` (tolerância
em caixa no Lab, um eixo por canal), que é o mesmo em 3,3× menos tempo.

## Instalação no aparelho

```sh
./deploy.sh user@<ip-do-j5>       # precisa de ssh por WiFi
ssh user@<ip-do-j5>
sudo /opt/j5/gadget.sh add        # adiciona o mouse ao g1 e reata
sudo /opt/j5/gadget.sh status
```

O gadget e o `/dev/hidg0` são de root, então `gadget.sh` e `hidmouse.py` rodam
com `sudo`. O `deploy.sh` cuida da permissão de `/opt/j5`: se o usuário remoto não
puder criar o diretório, ele usa `sudo` só para o `mkdir` e passa a posse. Sem
`sudo` no aparelho, use `DEST=$HOME/j5 ./deploy.sh user@<ip>` e ajuste `J5_DIR`
em `/etc/conf.d/j5-hid`.

`gadget.sh add` **desata o gadget por ~1 s**: se o seu shell estiver em cima do
`usb0`, você se tranca pra fora. Use WiFi.

Se o host recusar ponteiro absoluto (raro: BIOS/UEFI, alguns NVR), troque o
descriptor:

```sh
/opt/j5/gadget.sh add --rel
/opt/j5/hidmouse.py --mode rel
```

O relativo desloca um **modelo** de posição a partir do que foi escrito; se um
mouse humano mexer no cursor no meio, o modelo deriva e é preciso reiniciar.

### Depois de todo reboot

O initramfs recria o `g1` do zero (só `ncm.usb0`), então o mouse some. Rode
`gadget.sh add` de novo, ou instale o serviço:

```sh
install -m 755 /opt/j5/j5-hid.initd /etc/init.d/j5-hid
rc-update add j5-hid default
rc-service j5-hid start
```

## Teste do atuador, sem visão nenhuma

Este é o teste que decide se o projeto existe. No PC:

```sh
python3 vision/sender.py --pattern circle --duration 8 --host <ip-do-j5>
```

O cursor deve desenhar um círculo. Se desenhar, o resto é só visão. Se não,
`hidmouse.py --dry-run` no aparelho mostra se os pacotes chegam, e
`gadget.sh status` mostra se o `/dev/hidg0` existe.

Antes de tudo isso, teste o caminho inteiro sem o aparelho, em loopback:

```sh
python3 hidmouse.py --dry-run --duration 6 --stats &
python3 vision/sender.py --pattern sweep --duration 5
```

## Visão

### Caminho rápido: captura de tela (X11)

```sh
python3 vision/run.py --source x11 --color d000d0 --host <ip-do-j5> --preview
```

Wayland devolve preto por XGetImage — o código recusa em vez de fingir que
detectou nada. Entre em "GNOME on Xorg".

## Sem câmera nenhuma

Existem quatro jeitos de rodar sem hardware de imagem, em ordem de fidelidade
decrescente para o desenvolvimento:

```sh
# 1. a fonte real e mais barata: a própria tela
python3 vision/run.py --source x11 --color d000d0 --host <ip>

# 2. sintético: um quadrado que anda, sem captura nenhuma
python3 vision/run.py --source test --color d000d0 --rect 0,0,1920,1080 --host <ip>

# 3. gravou uma vez, ajusta para sempre (pacing original, jitter incluído)
python3 vision/record.py --source http --url http://192.168.42.1:8080 --hz 20 --duration 20 --out rec
python3 vision/run.py --source replay --replay-dir rec --model color.json --calib calib.json --host <ip>

# 4. uma câmera falsa de verdade, no caminho v4l2 real
sudo dnf install v4l2loopback && sudo modprobe v4l2loopback video_nr=10 exclusive_caps=1
ffmpeg -f x11grab -framerate 30 -video_size 1280x720 -i :0.0+0,0 -pix_fmt yuv420p -f v4l2 /dev/video10
python3 vision/run.py --source v4l2 --device /dev/video10 --model color.json --calib calib.json --host <ip>
```

O item 4 é o mais próximo de "emular um dispositivo de captura": `/dev/video10` é
um dispositivo V4L2 legítimo que qualquer biblioteca abre como câmera, e o
`--source v4l2` usa `CAP_PROP_BUFFERSIZE=1` para não herdar a fila do driver
(com a fila padrão o `read()` volta instantâneo com um frame de 3 a 5 períodos
atrás, e você caça latência fantasma por uma tarde).

O item 3 é o que realmente importa no dia a dia: gravar 20 s de câmera uma vez e
ajustar threshold, calibração e filtro offline, de forma reproduzível. `record.py`
grava `meta.json` com a região da tela, então um replay de captura de tela
dispensa `--rect`.

### E o J5 como dispositivo de captura?

Dá, e sem rebuild: `CONFIG_USB_CONFIGFS_F_UVC=y` está no kernel do pmOS, então o
J5 pode se apresentar como webcam UVC e o PC vê um `/dev/videoN` normal. Mas
**os pixels têm que vir de algum lugar**, e o único lugar possível é o próprio PC
— a câmera do aparelho não existe no mainline. O resultado é um laço: tela do PC
→ USB → J5 → USB → PC como câmera.

Serve como banco de testes honesto (negociação de formato, descarte de frame,
timestamps, o caminho `v4l2` inteiro) e é mais trabalhoso que o item 4, que faz o
mesmo sem o aparelho. O que ele **não** faz é deixar o J5 autônomo.

Para o J5 ser de fato o "olho", ele precisaria receber a tela como display, e aí
o caminho é o gadget GUD — que não é mainline (precisa de patch no kernel) e o
driver `DRM_GUD` está desligado no config do pmOS. Mesmo depois disso, a
detecção num A53 de 1,2 GHz com 1,5 GB de RAM rodaria a ~10 fps em 320×240,
contra 2 ms no PC.

## Caminho câmera: celular na rede ou no tether

O app de câmera IP precisa servir um snapshot JPEG (`/shot.jpg` no IP Webcam).
Polling de snapshot ganha do stream MJPEG em latência: o stream tem buffer entre
o sensor e você, então pode te entregar um frame de 300 ms atrás com um `read`
instantâneo.

```sh
curl -s http://192.168.42.1:8080/shot.jpg | file -     # tem que dizer JPEG
```

Calibração (a câmera não sabe onde o pixel cai):

```sh
# 1. no navegador: trainer.html, tecla 5, espera os 9 pontos, tecla e -> JSON
# 2. grava o mesmo período com a câmera
python3 vision/record.py --source http --url http://192.168.42.1:8080 \
    --hz 20 --duration 15 --out rec-calib
# 3. pareia por relógio de parede e ajusta
python3 vision/calibrate.py --from rec-calib --trainer-json ~/Downloads/j5-trainer-calib-*.json
```

Sai `calib.json` com o modelo escolhido por erro de reprojeção
(`poly2` costuma ganhar do `homography` porque absorve a distorção da lente) e
o rms. Acima de 3 px é suspeita — recalibre com a câmera mais de frente.

```sh
python3 vision/run.py --source http --url http://192.168.42.1:8080 \
    --model color.json --calib calib.json --rect 0,0,1920,1080 --preview
```

## Trainer

`trainer.html` abre sozinho no navegador. Serve para medir e para calibrar.

| Tecla | Ação |
|---|---|
| `1`–`5` | flick, grid, track, chase, calib |
| `r` | reset |
| `f` | fullscreen |
| `e` | exporta JSON |

O modo `chase` mede a **latência do loop inteiro**: um alvo percorre uma
circunferência a 0,5 Hz, a página mede a defasagem entre o cursor e o alvo e
converte o ângulo em milissegundos. É o único número honesto, porque inclui a
câmera.

Parâmetros por URL: `?color=d000d0&size=72&dwell=120&mode=chase`.

Cor recomendada: **magenta** (`d000d0`). Vermelho puro satura o sensor e o
chroma subsampling 4:2:0 do JPEG destrói as bordas; magenta sobrevive e não
colide com pele nem com UI.

## Formato na rede

10 bytes por datagrama UDP, porta 5555:

```
A5 5A | flags | x u16 | y u16 | age | seq | crc8
```

`age` é a idade do frame em ms: o atuador descarta o que estiver velho em vez de
perseguir um alvo de 300 ms atrás. `seq` descarta reordenação, `crc8` (polinômio
0x07) descarta corrupção. O range `0..32767` cobre o **desktop virtual inteiro**,
não um monitor — em multi-monitor calibre e escale contra a união de todos.

## O que mora no J5 e não no PC

O `hidmouse.py` aplica filtro 1-euro (segue rápido, não treme parado), teto de
passo por report, deadband e watchdog. Isso é deliberado: o último estágio antes
do HID é quem deve decidir *como* o cursor se move, e assim uma detecção
espúria vira um tranco pequeno em vez de um teleporte.

## Arquivos

```
protocol.py        formato do pacote (usado pelos dois lados)
report_desc.py     descriptor HID binário -> configfs
gadget.sh          add/del/status do mouse dentro do g1
hidmouse.py        atuador: UDP -> /dev/hidg0
j5-hid.initd       OpenRC, reproduz o add a cada boot
deploy.sh          scp pro aparelho (inclui o initd)
selftest.py        43 checagens, sem hardware
trainer.html       alvos, métricas e calibração
vision/source.py   x11 | http | mjpeg | v4l2 | replay | test
vision/detect.py   tolerância em caixa no Lab, rejeição de banda de rolling shutter
vision/calibrate.py poly2 vs homografia, escolhido por erro medido
vision/sender.py   UDP + padrões de teste
vision/run.py      o loop
vision/record.py   gravação com timestamp (calibração e depuração)
```

## Ajustes que existem por um motivo

- `hidmouse.py` descarta o report em `EAGAIN` em vez de enfileirar: uma fila de
  reports velhos é pior que um frame perdido.
- O deadband não se aplica ao **primeiro** report depois de uma pausa. Sem isso,
  mirar no centro exato da tela logo depois de iniciar não move nada, porque o
  atuador acha que já está lá.
- `gadget.sh` tem um `trap` que reata o UDC se morrer entre o unbind e o bind.
  Sem ele, um `add` que falha deixa o `usb0` morto até o próximo boot.
- `vision/record.py` grava `meta.json` com a região da tela, e é isso que permite
  dar replay numa captura de tela sem passar `--rect`.
- `--source v4l2` força `CAP_PROP_BUFFERSIZE=1`: com a fila padrão o `read()`
  devolve na hora um frame de 3 a 5 períodos atrás.