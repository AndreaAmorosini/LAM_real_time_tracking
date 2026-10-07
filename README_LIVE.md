# LAM — Live Tracking con frontend WebGL

Questa guida descrive l'installazione e l'uso della variante live di LAM su Linux. Da una foto viene generato un avatar; la webcam collegata alla macchina che esegue il backend fornisce il tracking facciale tramite MediaPipe; il browser riceve i dati via WebSocket e anima l'avatar WebGL.


## Requisiti

- Linux `x86_64`.
- GPU NVIDIA con driver compatibile con CUDA 12.1.
- Webcam accessibile **dalla macchina backend**.
- Connessione Internet per dipendenze, modelli e asset.
- Spazio su disco per checkpoint LAM, componenti CUDA, Blender e frontend.
- [Pixi](https://pixi.sh/) installato e disponibile nel `PATH`.

Le dipendenze di sistema suggerite sono:

```bash
sudo apt update
sudo apt install -y git curl wget build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libegl1 v4l-utils
```

Se Pixi non è installato, seguire le istruzioni ufficiali di Pixi. Una possibile installazione è:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Aprire un nuovo terminale, oppure rendere disponibile Pixi nella sessione corrente:

```bash
export PATH="$HOME/.pixi/bin:$PATH"
```

> I comandi che seguono vanno eseguiti **dalla directory principale della repository**.

## Installazione

### Installazione tramite task aggregato

```bash
pixi install
pixi run setup-all
```

`setup-all` installa le dipendenze Python e PyTorch, compila le estensioni, scarica i pesi LAM e il modello MediaPipe, installa Blender e l'SDK FBX, prepara il frontend WebGL ed esegue i controlli `doctor`. Le operazioni possono richiedere molto tempo. I log dello script vengono scritti in `logs/setup/`.

Lo script non scarica esplicitamente gli asset di esempio OpenAvatarChat richiesti dall'esportatore. Prima di creare un avatar, verificare che esistano:

```text
assets/sample_oac/template_file.fbx
assets/sample_oac/animation.glb
```

Se mancano, scaricarli dalla sorgente usata dal progetto:

```bash
wget -O /tmp/sample_oac.tar \
  https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/aigc3d/data/LAM/sample_oac.tar
tar -xf /tmp/sample_oac.tar -C assets/
rm /tmp/sample_oac.tar
```

### Installazione per fasi

In alternativa a `setup-all`, eseguire i task nell'ordine seguente:

```bash
pixi install
pixi run setup-python-base
pixi run setup-torch-cu121
pixi run setup-live
pixi run setup-weights
pixi run setup-faceboxes
pixi run setup-cuda-ext
pixi run setup-mediapipe-model
pixi run setup-blender
pixi run setup-fbx-sdk
pixi run setup-webgl
pixi run webgl-build
pixi run doctor
```

Scaricare inoltre `sample_oac.tar` come descritto sopra, se i relativi file non sono già presenti.

I nomi dei task sono quelli definiti in `pixi.toml`. Per esempio, il task del frontend è `setup-webgl`, **non** `webgl-install`.

### Architettura CUDA

Lo script di compilazione delle estensioni imposta come valore predefinito:

```bash
TORCH_CUDA_ARCH_LIST=8.9
```

Se la GPU usa un'architettura diversa, impostare la variabile per la propria GPU **prima** di compilare le estensioni:

```bash
export TORCH_CUDA_ARCH_LIST="<architettura_della_tua_GPU>"
pixi run setup-cuda-ext
```

## Verifiche

Controllare l'ambiente e la disponibilità della GPU:

```bash
pixi run doctor
pixi run check-gpu
pixi run check-blender
```

Sono disponibili anche comandi diagnostici che accedono alla webcam o al tracking:

```bash
pixi run test-webcam
pixi run test-mediapipe
pixi run test-live-motion
pixi run test-flame-adapter
```

> Questi task sono principalmente prove manuali o diagnostiche, non una suite di test automatizzati. `test-webcam` e `test-mediapipe` richiedono una webcam funzionante.

Se la webcam non è accessibile, controllare i dispositivi:

```bash
v4l2-ctl --list-devices
```

Verificare inoltre i permessi dell'utente sul dispositivo video. Un'eventuale aggiunta al gruppo `video` richiede una nuova sessione di login:

```bash
sudo usermod -aG video "$USER"
```

## Avvio

Dopo aver costruito il frontend:

```bash
pixi run app-live-web
```

Aprire:

```text
http://127.0.0.1:7861/
```

La pagina WebGL viene servita **alla root `/`**, non a `/webgl/`. L'applicazione viene avviata da `app_live_web.py` con host `127.0.0.1` e porta `7861`.

### Flusso d'uso

1. Aprire la pagina nel browser.
2. Selezionare una foto frontale del volto.
3. Lasciare **disattivata** l'opzione multiview.
4. Avviare la creazione dell'avatar dall'interfaccia.
5. Attendere preprocessing, generazione LAM ed esportazione del pacchetto WebGL.
6. Una volta caricato l'avatar, il frontend si collega al WebSocket e applica il tracking ricevuto dal backend.
7. Usare il comando di arresto dell'interfaccia per fermare il tracking e scaricare l'avatar.

L'esportazione genera uno ZIP contenente, fra gli altri:

```text
offset.ply
skin.glb
animation.glb
vertex_order.json
```

Il browser **non** invia il video della propria webcam: l'acquisizione avviene sul computer su cui gira Python. Anche aprendo la pagina da un altro dispositivo, la webcam utilizzata rimane quella del backend.

## Endpoint principali

| Endpoint | Funzione |
| --- | --- |
| `GET /` | Frontend WebGL compilato |
| `POST /api/oac/export` | Esportazione avatar da una foto |
| `GET /api/client-config` | Parametri inviati al frontend |
| `WS /ws/live` | Tracking e dati di animazione in tempo reale |
| `GET /oac_assets/<file>` | Asset esportati |
| `POST /api/cleanup/export` | Pulizia degli export temporanei registrati |
| `POST /api/oac/export-multiview` | Percorso sperimentale, attualmente non funzionante senza modifiche |

Nel flusso WebGL il frontend apre `/ws/live` con `output_mode=webgl`. Il backend prevede anche `output_mode=debug` e `output_mode=lam` per client WebSocket dedicati; **non** esiste, in questa versione, una pagina di debug backend separata alla root. I controlli di debug presenti nella pagina WebGL possono richiedere immagini dei landmark e informazioni sui parametri FLAME.

## Directory generate

Durante l'uso vengono prodotti file principalmente in:

```text
output/open_avatar_chat/
output/live_uploads/
tracking_output_live/
```

La cartella del frontend compilato è:

```text
webgl_frontend/dist/
```

Queste directory non vanno confuse con i file sorgente del progetto. Alcune esportazioni possono essere riutilizzate dalla cache in base al contenuto della foto.

## Configurazione e limiti

Le impostazioni del backend sono definite in `lam/live/settings.py`; la classe legge anche variabili d'ambiente e un eventuale file `.env` nella root. Fra le opzioni disponibili figurano percorsi degli output, dimensioni di acquisizione, FPS e percorso di Blender.

Non tutte le impostazioni dichiarate sono applicate dall'avvio attuale:

- `app_live_web.py` avvia Uvicorn su `127.0.0.1:7861` con valori espliciti; `LAM_WEB_HOST`, `LAM_WEB_PORT` e `LAM_WEB_RELOAD` non modificano tale avvio.
- Il WebSocket istanzia `LiveMotionProvider` senza passargli `settings.camera_index`: la webcam predefinita resta il dispositivo `0`.
- La pagina viene montata su `/`, indipendentemente dal valore dichiarato per `webgl_route`.

L'esportazione da una singola foto accetta nell'interfaccia anche un percorso personalizzato di Blender; se lasciato vuoto, il backend usa il proprio percorso predefinito. Il task `app-live-web` imposta automaticamente `LAM_BLENDER_PATH` verso Blender installato in `thirdparties/blender/`.

## Sviluppo del frontend

Per lavorare sul frontend con Vite, avviare il backend e, in un altro terminale, il server di sviluppo:

```bash
pixi run app-live-web
```

```bash
pixi run webgl-dev
```

Aprire quindi:

```text
http://127.0.0.1:5173/
```

La configurazione Vite inoltra `/api`, `/ws` e `/oac_assets` al backend locale sulla porta `7861`. Per ricostruire la versione servita dal backend:

```bash
pixi run webgl-build
```

## Risoluzione dei problemi

### `pixi run doctor` fallisce

Controllare il messaggio specifico: `doctor` verifica import Python, Blender, directory del frontend e presenza dei modelli. Non sostituisce una prova completa di generazione dell'avatar.

### CUDA o compilazione delle estensioni falliscono

Verificare il driver NVIDIA, `pixi run check-gpu` e l'architettura impostata in `TORCH_CUDA_ARCH_LIST`. La compilazione installa anche dipendenze native da repository Git esterni.

### L'avatar non viene esportato

Verificare checkpoint LAM, asset in `assets/sample_oac/`, Blender e SDK FBX. Controllare inoltre l'errore restituito dall'API `/api/oac/export`.

### La pagina si apre, ma il tracking non parte

La webcam deve essere collegata alla macchina backend ed essere disponibile come dispositivo `0` nell'implementazione corrente. Controllare i permessi, `pixi run test-webcam`, `pixi run test-mediapipe` e la console del backend.

### La modalità multiview fallisce

È un limite noto della versione corrente: `app_live_web.py` usa diverse proprietà `settings.multiview_*` non definite in `lam/live/settings.py`. Utilizzare l'esportazione da una sola foto finché il percorso multiview non viene completato.

## Nota di sicurezza

L'avvio predefinito è limitato a `127.0.0.1`. Non esporre direttamente il servizio a reti non fidate senza una revisione di autenticazione, upload e pulizia dei file. In particolare, la funzione `safe_remove_path()` in `lam/live/cleanup.py` segnala i percorsi fuori dalle directory consentite, ma nella versione attuale non interrompe la rimozione: va corretta prima di un'esposizione di rete.