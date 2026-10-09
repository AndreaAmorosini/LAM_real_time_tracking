# LAM — Live Tracking con frontend WebGL e Studio

Questa guida descrive l'installazione e l'uso della variante live di LAM su Linux. Da una foto viene generato un avatar; la webcam collegata alla macchina che esegue il backend fornisce il tracking facciale tramite MediaPipe; il browser riceve i dati via WebSocket e anima l'avatar WebGL. Sono disponibili il frontend originale (`webgl_frontend/`) e la UI alternativa Studio (`webgl_frontend_alt/`), che condividono API, modelli e tracking.


## Requisiti

- Linux `x86_64`.
- GPU NVIDIA con driver compatibile con CUDA 12.1.
- Webcam accessibile **dalla macchina backend**.
- Per scattare foto dalla UI Studio serve anche una webcam accessibile dal **browser**; su dispositivi remoti `getUserMedia` richiede HTTPS, mentre `localhost` è consentito come contesto sicuro.
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

`setup-all` installa le dipendenze Python e PyTorch, compila le estensioni, scarica i pesi LAM e il modello MediaPipe, installa Blender e l'SDK FBX, costruisce il frontend originale e Studio ed esegue i controlli `doctor`. Le operazioni possono richiedere molto tempo. I log dello script vengono scritti in `logs/setup/`.

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
pixi run studio-build
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

### UI Studio (consigliata)

```bash
pixi run studio-build
pixi run app-live-studio
```

Aprire `http://studio.localhost:7861/`. Con lo **stesso processo** si può aprire `http://localhost:7861/` per il frontend originale. Il server seleziona la pagina tramite l'header `Host`: su `studio.localhost` serve Studio, sugli altri host la UI originale; gli endpoint `/api`, `/ws` e `/oac_assets` restano condivisi. `app-live-studio` ascolta solo su `127.0.0.1:7861` e sostituisce `app-live-web` sulla stessa porta: non avviare entrambi contemporaneamente.

Il nome dell'host Studio è configurabile **prima** di avviare il server:

```bash
export LAM_STUDIO_HOST=studio.localhost
pixi run app-live-studio
```

Per usare un sottodominio pubblico servono anche DNS, HTTPS e un reverse proxy che preservi l'header `Host` e inoltri HTTP e WebSocket al backend; impostare `LAM_STUDIO_HOST` da solo non pubblica il servizio. Leggere prima la [nota di sicurezza](#nota-di-sicurezza).

### UI originale

Se si desidera avviare soltanto il frontend esistente:

```bash
pixi run webgl-build
pixi run app-live-web
```

Aprire:

```text
http://127.0.0.1:7861/
```

La pagina WebGL viene servita **alla root `/`**, non a `/webgl/`. `app-live-web` avvia `app_live_web.py` su `127.0.0.1:7861`; in questa modalità non viene servita la UI Studio.

### Guida all'uso di Studio

1. Aprire `http://studio.localhost:7861/` e scegliere **Foto singola** oppure **Multi-view**.
2. Caricare una foto frontale dal disco, oppure premere **Usa webcam guidata**. Prima di aprire la webcam del browser appare un'informativa: selezionare la checkbox e premere **Continua alla webcam**; senza consenso non viene richiesto l'accesso alla webcam.
3. Con **Foto singola**, centrare il viso nella guida trasparente e scattare. Con **Multi-view**, scattare in sequenza frontale, destra, sinistra, alto e basso seguendo la guida sovrapposta al video. In alternativa caricare le immagini nei rispettivi campi; il frontale è obbligatorio, le altre viste sono opzionali. Si possono regolare le iterazioni di raffinamento.
4. Dopo l'ultimo scatto parte la generazione; se si usano file dal disco, premere **Genera avatar**. La rotella mostra messaggi **generici**, non una percentuale di avanzamento reale: attendere preprocessing, esportazione e caricamento WebGL senza chiudere la pagina.
5. Una volta caricato l'avatar, il tracking della webcam **del backend** anima l'avatar. Si può cambiare la mappatura tra **Stabile**, **Espressiva** e **Raw / debug**; in **Controlli avanzati** sono disponibili percorso Blender opzionale, visualizzazioni MediaPipe/FLAME e override di blendshape e ossa.
6. Dopo il caricamento compare **Espressioni** con sei pulsanti: **Smile**, **Prohibited**, **Concerned**, **Thoughtful**, **Disgusted** e **Wink**. Ognuno applica un preset di blendshape con breve transizione e sospende la connessione WebSocket, fermando il tracking MediaPipe. Sono pose statiche, non clip di animazione temporizzate. **Riprendi tracking live** riapre la connessione e torna alla webcam del backend.
7. Premere **Ferma tracking e rimuovi avatar** per chiudere la sessione e rimuovere l'avatar dalla pagina. Il pulsante non scarica automaticamente un file sul computer: lo ZIP esportato si trova sul server nella directory indicata sotto.

Il percorso Multi-view è sperimentale e richiede più tempo e memoria del percorso a foto singola; non è garantito che completi la ricostruzione su ogni configurazione. Per una prima prova usare **Foto singola**.

### Uso della UI originale

Caricare una foto frontale oppure usare **Attiva fotocamera** per acquisire una singola foto dal browser, poi avviare la generazione. L'opzione Multi-view accetta file separati ma non offre la guida sovrapposta alla webcam di Studio. Al termine il browser riceve i dati live da `/ws/live`; sono disponibili mappatura e controlli di debug. **Stop Tracking + Unload Avatar** rimuove l'avatar dalla pagina, senza scaricare lo ZIP.

L'esportazione genera uno ZIP contenente, fra gli altri:

```text
offset.ply
skin.glb
animation.glb
vertex_order.json
```

Il browser non invia uno stream video della propria webcam al backend: quando si usa la funzione di scatto, invia invece le **foto acquisite** all'API di esportazione. Il tracking continuo avviene sul computer su cui gira Python. Anche aprendo la pagina da un altro dispositivo, il tracking utilizza la webcam del backend.

## Endpoint principali

| Endpoint | Funzione |
| --- | --- |
| `GET /` | UI Studio sull'host `studio.localhost` con `app-live-studio`; UI originale sugli altri host o con `app-live-web` |
| `POST /api/oac/export` | Esportazione avatar da una foto |
| `GET /api/client-config` | Parametri inviati al frontend |
| `WS /ws/live` | Tracking e dati di animazione in tempo reale |
| `GET /oac_assets/<file>` | Asset esportati |
| `POST /api/cleanup/export` | Pulizia degli export temporanei registrati |
| `POST /api/oac/export-multiview` | Esportazione Multi-view sperimentale |

Nel flusso WebGL il frontend apre `/ws/live` con `output_mode=webgl`. Il backend prevede anche `output_mode=debug` e `output_mode=lam` per client WebSocket dedicati; **non** esiste, in questa versione, una pagina di debug backend separata alla root. I controlli di debug presenti nella pagina WebGL possono richiedere immagini dei landmark e informazioni sui parametri FLAME.

## Directory generate

Durante l'uso vengono prodotti file principalmente in:

```text
output/open_avatar_chat/
output/live_uploads/
tracking_output_live/
```

Le cartelle dei frontend compilati sono:

```text
webgl_frontend/dist/
webgl_frontend_alt/dist/
```

Queste directory non vanno confuse con i file sorgente del progetto. Alcune esportazioni single-photo possono essere riutilizzate dalla cache in base al contenuto della foto.

### Conservazione dei dati

La webcam del browser viene usata per l'anteprima e lo scatto, senza registrare un video continuo nell'app. Le foto scattate sono inviate al backend per la generazione. Il backend salva lo ZIP single-photo in `output/open_avatar_chat/` come cache e non lo elimina quando si preme **Ferma tracking e rimuovi avatar**. Per Multi-view, lo ZIP temporaneo viene registrato per la pulizia alla chiusura della sessione (operazione best effort); possono comunque restare input intermedi in `tracking_output_live/raw_inputs/`, file di diagnostica o export incompleti dopo errori. **L'applicazione non garantisce l'eliminazione di tutti i dati personali**: non interpretare la frase sulla cancellazione nell'informativa della UI Studio come una garanzia implementata dal backend.

## Configurazione e limiti

Le impostazioni del backend sono definite in `lam/live/settings.py`; la classe legge anche variabili d'ambiente e un eventuale file `.env` nella root. Fra le opzioni disponibili figurano percorsi degli output, dimensioni di acquisizione, FPS e percorso di Blender.

Non tutte le impostazioni dichiarate sono applicate dall'avvio attuale:

- `app_live_web.py` e `app_live_studio.py` avviano Uvicorn su `127.0.0.1:7861` con valori espliciti; `LAM_WEB_HOST`, `LAM_WEB_PORT` e `LAM_WEB_RELOAD` non modificano tale avvio. `LAM_STUDIO_HOST` modifica solo l'host usato per selezionare la UI Studio.
- Il WebSocket istanzia `LiveMotionProvider` senza passargli `settings.camera_index`: la webcam predefinita resta il dispositivo `0`.
- La pagina viene montata su `/`, indipendentemente dal valore dichiarato per `webgl_route`.

L'esportazione da una singola foto accetta nell'interfaccia anche un percorso personalizzato di Blender; se lasciato vuoto, il backend usa il proprio percorso predefinito. I task `app-live-web` e `app-live-studio` impostano automaticamente `LAM_BLENDER_PATH` verso Blender installato in `thirdparties/blender/`.

## Sviluppo del frontend

Per lavorare sul frontend con Vite, avviare il backend e, in un altro terminale, il server di sviluppo della UI desiderata:

```bash
pixi run app-live-studio
```

Per Studio:

```bash
pixi run studio-dev
```

Aprire `http://127.0.0.1:5174/` (server Vite, senza instradamento per sottodominio). Per la UI originale:

```bash
pixi run webgl-dev
```

Aprire `http://127.0.0.1:5173/`. I due server Vite possono condividere il backend sulla porta `7861`. Per ricostruire le versioni servite dal backend:

```bash
pixi run webgl-build
pixi run studio-build
```

Le configurazioni Vite inoltrano `/api`, `/ws` e `/oac_assets` al backend locale sulla porta `7861`.

## Risoluzione dei problemi

### `pixi run doctor` fallisce

Controllare il messaggio specifico: `doctor` verifica import Python, Blender, directory del frontend e presenza dei modelli. Non sostituisce una prova completa di generazione dell'avatar.

### CUDA o compilazione delle estensioni falliscono

Verificare il driver NVIDIA, `pixi run check-gpu` e l'architettura impostata in `TORCH_CUDA_ARCH_LIST`. La compilazione installa anche dipendenze native da repository Git esterni.

### L'avatar non viene esportato

Verificare checkpoint LAM, asset in `assets/sample_oac/`, Blender e SDK FBX. Controllare inoltre l'errore restituito dall'API `/api/oac/export`.

### La pagina si apre, ma il tracking non parte

La webcam deve essere collegata alla macchina backend ed essere disponibile come dispositivo `0` nell'implementazione corrente. Controllare i permessi, `pixi run test-webcam`, `pixi run test-mediapipe` e la console del backend.

### La webcam guidata non si apre

Verificare di aver selezionato la checkbox nell'informativa, concesso il permesso webcam al browser e aperto la pagina su `localhost`, un sottodominio `.localhost` oppure su un'origine HTTPS. Questa webcam serve allo scatto delle foto, non sostituisce la webcam del backend per il tracking live.

### Studio mostra la UI originale o una pagina vuota

Verificare di aver eseguito `pixi run studio-build`, avviato `pixi run app-live-studio` (non `app-live-web`) e aperto `http://studio.localhost:7861/`. Se `studio.localhost` non si risolve sulla macchina, configurare un hostname locale che punti a `127.0.0.1` e impostare `LAM_STUDIO_HOST` di conseguenza prima dell'avvio.

### La modalità multiview fallisce

È un percorso sperimentale: le proprietà `settings.multiview_*` sono ora presenti, ma ciò non equivale a una verifica end-to-end dell'esportazione. Controllare l'errore restituito da `/api/oac/export-multiview`, le foto fornite, la GPU e la memoria disponibili; usare la foto singola se il raffinamento fallisce.

## Nota di sicurezza

L'avvio predefinito è limitato a `127.0.0.1`. Non esporre direttamente il servizio a reti non fidate senza una revisione di autenticazione, upload e pulizia dei file. In particolare, la funzione `safe_remove_path()` in `lam/live/cleanup.py` segnala i percorsi fuori dalle directory consentite, ma nella versione attuale non interrompe la rimozione: va corretta prima di un'esposizione di rete.
