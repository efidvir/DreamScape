# DreamScape

A voice-driven lucid dream. You talk to a being that has just woken up in black mist and knows nothing about itself. Whatever you tell it becomes true: it becomes a fairy, the mist becomes a dark forest, it builds a spaceship, flies to the moon, finds a block of cheese… Every turn the dream grows on screen in live 3D, drawn entirely in soft particles: things gather out of the fog as glowing dust, sharpen for a moment, and drift back into haze, like a dream half remembered.

Everything runs locally on one GPU laptop, as a set of Docker containers that report their heartbeats to each other. Hebrew and English both work: speech in, speech out, and the story itself.

## How a turn works

```mermaid
flowchart LR
  B["Browser<br/>3D dream, mic, voice"] <-- WebSocket --> W["web<br/>page, turn pipeline, heartbeats"]
  W -- "recorded speech" --> V["voice<br/>Whisper STT · BlueTTS"]
  W -- "player's words" --> D["director<br/>story memory, scene"]
  D -- "chat + JSON schema" --> O["ollama<br/>Gemma 4 E4B"]
  W -. "optional" .-> I["imagery<br/>Dask scheduler"]
  I -. "Dask" .-> G["GPU worker<br/>second machine"]
```

1. You hold the glowing orb (or the space bar) and speak. The browser sends the recording to **web**.
2. **voice** transcribes it with ivrit.ai's Hebrew-tuned Whisper.
3. **director** asks the language model for one JSON answer that holds both the dreamer's spoken reply and the scene changes (new setting, things that appear, effects). Ollama's structured outputs guarantee valid JSON, and the director keeps the story's facts and the scene between turns. The model writes the reply early in its answer, and the director streams its first sentence out the moment that sentence is complete. If those first words come out in the wrong language, the director asks again before you hear anything.
4. **voice** starts speaking that first sentence with BlueTTS (a long one is cut at its first comma) while the model is still writing the rest, so the dreamer starts talking about 2 to 3 seconds after you press Enter, or 2.5 to 4 seconds after you stop speaking.
5. The scene update follows a second or two later and materialises while the dreamer talks; its light pulses with its voice.

### How the dream is drawn

Everything on screen is a particle cloud. The renderer knows 13 settings (void, forest, meadow, desert, ocean, mountains, cave, city, space, moon, clouds, snow, underwater) and 39 kinds of things (fairy, dragon, unicorn, castle, spaceship, cheese, portal, crystals…), each with colour, size, count, placement and motion. Each one is shaped in 3D, then sampled into thousands of soft particles. Its hard geometry is never drawn.

- **Appearing and vanishing:** particles stream in from the surrounding mist when a thing appears and scatter back when it goes. When the dreamer changes form, one cloud dissolves while the next gathers in the same place.
- **Haze:** the dream breathes between hazy and clear. In the haze, particles drift loose and shapes soften, and some dust always floats off the edges.
- **Focus:** each particle blurs by its own distance from the point of attention, like a camera lens. The lens follows whatever just appeared.
- **Pictures (with the imagery service on):** every new thing and every new form of the dreamer also gets a generated picture. The picture is sampled into its own particle cloud, which takes over from the geometric one, so the dream looks painted.

Anything outside the vocabulary appears as a shimmering apparition until its picture arrives.

## Run it

You need Docker Desktop (WSL 2 backend), an NVIDIA driver with WSL support, and about 15 GB of free disk for images and models.

On Windows, double-click **start.bat**: it builds and starts every container, then opens the page once it answers. From a terminal:

```powershell
copy .env.example .env        # optional: every setting has a default
docker compose up -d --build
```

Open **http://localhost**. The first start downloads the models (Gemma 4 E4B ≈ 6.6 GB, Whisper turbo ≈ 1.6 GB, and the BlueTTS bundle ≈ 0.7 GB). Until they are ready, the page lists what is still loading with its progress, and anything you say or type waits and is answered as soon as the storyteller is ready. The dots in the top corner show each service's heartbeat; hover for details such as tokens per second.

After every start the storyteller runs one short warm-up turn, exactly like a real one, so your first turn is as quick as the rest. The first start after the Ollama container is created also compiles GPU code, which makes the first ten minutes or so of play slower; the compiled code is kept in the `ollama` volume for every start after that.

Browsers allow the microphone only on `localhost` or over HTTPS, so speak to the dream from the laptop itself; from another device on the network you can still type. Press F2 to see how long each turn took (speech-to-text, storyteller, voice).

### Live code and checking a running game

`docker-compose.override.yml` runs the web, storyteller and voice services straight from this folder and restarts each one when its code changes, so edits apply without rebuilding. Page files apply on the next page reload. Delete the file to run only the code baked into the images.

- **http://localhost/api/status**: every service's heartbeat, as the page sees it.
- **http://localhost/api/debug/recent**: the storyteller's last raw answers, and what the director filled in from the player's own words (see below).
- **http://localhost/api/debug/gpu**: video memory, load, clock, temperature and why the clock is held back, plus the processors' load.
- **http://localhost/api/debug/tts?text=...&lang=he**: how long one sentence takes to synthesize.
- **http://localhost/api/debug/director/models** and **/loaded**: the downloaded models with their sizes, and what is in memory now.
- `POST /api/debug/director/turn` with `{"session_id", "text", "model"}` answers one turn with any downloaded model and reports its speed; `POST /api/debug/director/model` with `{"model"}` plays with another model until the next restart (`LLM_MODEL` makes it stick); `/pull` downloads one alongside, `/unload` frees one.

### When the model misses something

A 4B model sometimes answers beautifully and still leaves the scene unchanged. After each turn the director reads the player's own words (Hebrew and English word lists in `services/director/app/lexicon.py`) and fills what the model left out: the place, what the dreamer is, its voice, weather, time of day and the things named. The model's own choices always win; the dream only moves to a new place when the player names one or travels. A turn the model fails twice comes back as a short notice, and the story stays unchanged, so the player can simply say it again. Saying the dreamer wakes up in bed ("אתה מתעורר במיטה", "you wake up in your bed") ends the dream. Words said while the dreamer is still answering are kept and answered right after, and if the storyteller restarts in the middle of a turn, the turn waits for it instead of failing.

### A gentle dream

Whatever the player says becomes true, so the dream keeps itself gentle without ever saying so. Swearing, slurs, sexual words and cruelty or gore dissolve before the storyteller hears them: the player's line on screen shows "…" where those words were, and the storyteller lets the dream drift somewhere softer. The same check runs over everything the dreamer says and over the scene, so its words, its name, the things in the dream and the stored story stay clean. Subtler requests (a cruel plan with no listed word in it, say) are handled by the storyteller's own instructions, and an answer that steps out of the dream ("as an AI…") becomes the picture blurring away. A player who speaks of hurting themselves keeps their words; the dream grows warm and the dreamer answers them with care. The word lists (Hebrew and English, with prefixes, stretched letters and look-alike spellings) are in `services/director/app/guard.py`.

Useful commands:

```powershell
docker compose logs -f director voice   # watch the storyteller and the voice load
docker compose ps                       # health of every container
docker compose down                     # stop (models and dreams stay in volumes)
```

### Video memory on the RTX 4070 Laptop (8 GB)

| Part | Runs on | Approx. |
|---|---|---|
| Gemma 4 E4B (Q4) with a 4K context | GPU | 4.7 GB |
| Whisper large-v3-turbo, ivrit.ai, int8 | GPU | 1.5 GB |
| BlueTTS | CPU, or GPU with `TTS_DEVICE=cuda` | none, or up to `TTS_GPU_MB` |
| The 3D dream in the browser | GPU | 0.3 GB |
| Background pictures | second machine | none |

Measured together on the RTX 4070 Laptop: 7.6 of 8.2 GB in use, so there is room for the game but not for another big GPU program next to it. `LLM_MODEL=gemma3:4b` in `.env` frees a gigabyte (6.6 GB in use) at the cost of plainer Hebrew and slower answers. If you already run Ollama on Windows, set `OLLAMA_URL=http://host.docker.internal:11434` in `.env` and the container uses it.

The dream page shares the graphics card with the storyteller, so while an answer is being written it draws 30 frames a second, and it keeps a full 60 the rest of the time.

## Settings (`.env`)

| Variable | Default | What it does |
|---|---|---|
| `WEB_PORT` | `80` | Port of the dream page |
| `LLM_MODEL` | `gemma4:e4b` | Any Ollama model tag; it is pulled automatically on first start. `gemma3:4b` is the lighter choice |
| `STT_MODEL` | `ivrit-ai/whisper-large-v3-turbo-ct2` | Speech-to-text model (faster-whisper) |
| `STT_MODEL_EN` | empty | A second model for English, loaded on the first English utterance |
| `VOICE_NEUTRAL` | `noa,ariel` | The dreamer's voice before it has a gender (two voices averaged) |
| `VOICE_FEMALE` / `VOICE_MALE` | `maya` / `daniel` | Voices once the story gives the dreamer a gender |
| `TTS_SPEED` | `0.95` | Speaking pace |
| `TTS_DEVICE` | `cpu` | `cuda` runs the voice on the GPU (it falls back to the CPU by itself if the GPU runtime cannot start). `POST /api/debug/voice/device` with `{"device": "cuda"}` or `"cpu"` switches a running game, to compare |
| `TTS_GPU_MB` | `900` | Video memory the GPU voice may use |
| `LLM_NUM_THREAD` | `0` | Processor threads for the parts of the model that run on the CPU (0: Ollama decides) |
| `COMPOSE_PROFILES` | empty | `imagery` turns on the picture service |
| `IMAGERY_URL` | empty | `http://imagery:9030` when the picture service is on |

Choose the page language at the bottom (Auto, עברית, English). In Auto the dreamer answers in whatever language you speak.

## Pictures from the second GPU machine (recommended)

The imagery service runs a Dask scheduler on the laptop. Your other GPU machine joins it as a worker and paints a picture with SD-Turbo for every new thing in the dream (plus a soft backdrop for each new place), so the laptop's video memory stays free. The page turns each picture into particles, so this is what gives the dream its painted look.

On the laptop, in `.env`:

```
COMPOSE_PROFILES=imagery
IMAGERY_URL=http://imagery:9030
```

On the second machine, copy the repository and run:

```powershell
cd worker
copy .env.example .env    # set SCHEDULER_HOST (laptop IP) and WORKER_HOST (this machine's IP)
docker compose up -d --build
```

The images dot turns green when the worker joins. Both machines build the same image, so their Dask versions always match. Port 8786 accepts work from anyone who can reach it, so keep it on your home network.

## Try it without a GPU

Scripted stand-ins for the language model, the voice and the picture painter let you run the real page and director anywhere. The stand-in painter only draws simple shapes; it shows how pictures become particles. Set `NO_PICTURES=1` to see the dream without it.

```powershell
pip install -r services/web/requirements.txt -r services/director/requirements.txt python-multipart pillow
python dev/demo.py              # then open http://localhost:8080
```

Say or type “you are a fairy in a dark forest”, “you build a spaceship and fly to the moon”, “there is a block of cheese”, “you eat it and become the god of the world”. `dev/fakes.py` lists more trigger words (dragon, castle, sea, cat, rain, snow…). `python dev/shots.py` drives that story in a headless browser and saves screenshots (first `pip install playwright` and `playwright install chromium`); `python dev/gallery.py` builds every thing and place and reports any errors.

## Project layout

```
services/
  web/        page server, WebSocket turn pipeline, heartbeat monitor
    static/js/dream/   Three.js dream: renderer, world, settings, things
  director/   game master over Ollama: prompt, JSON schema, story state (+ tests)
  voice/      faster-whisper STT and BlueTTS TTS
  imagery/    SD-Turbo pictures through Dask (one image for scheduler and worker)
worker/       compose file for the second GPU machine
dev/          GPU-free demo, fakes and screenshot script
start.bat     Windows: start everything and open the page
docker-compose.override.yml   live code from this folder (see "Live code")
```

Run the director's tests with `cd services/director && pip install -r requirements-dev.txt && pytest`.

## What changed from the first version

- The dream is drawn live in the browser with Three.js as soft particle clouds, with generated pictures turned into particles when the imagery service runs. The first version generated a video clip on every turn, which was the slow part.
- One language-model call per turn returns both the reply and the scene, as schema-checked JSON, with the story's facts carried between turns. TinyLlama gave way to Gemma 4 E4B, which tells a coherent story in natural Hebrew and English (Gemma 3 4B remains a lighter option).
- Speech is fully local: Whisper fine-tuned for Hebrew by ivrit.ai, and BlueTTS for Hebrew and English voices. The dreamer starts with a neutral voice and takes on a gendered one when the story decides.
- Every service reports a heartbeat; the web service gathers them and shows them on the page, and Docker health checks watch each container.
- Image generation moved to a separate, optional service that keeps the Dask route to the second machine.
- The game says plainly what is still loading and keeps the player's words until it is ready, instead of answering in character that it heard nothing. Failed turns never enter the story, and dreams saved with old fallback lines are cleaned when they load.
- Recordings are decoded with PyAV directly: faster-whisper's own decoder breaks on current PyAV releases.
- The dreamer's words stream out of the language model as soon as they are written, and its first sentence is voiced on its own, which halves the wait before it speaks.
- The storyteller is Gemma 4 E4B. On the same seven stories it wrote more natural Hebrew than Gemma 3 4B and answered faster (53 to 64 tokens a second against 30 to 44); Gemma 3 4B stays one line away in `.env`.
- The dreamer starts talking as soon as its first sentence is written, with a long first sentence cut at its first comma: about 2 to 3 seconds after a typed line.
- The model reads its worked examples with the player's language last, an answer in the wrong language is asked again before anyone hears it, Hebrew answers agree with the dreamer's gender, and the model lists only the facts each line adds.
- While an answer is being written, the dream page draws at half rate; at full rate it shared the graphics card so heavily that the storyteller slowed to a crawl after a few turns.
- The dreamer is easier to see: pale colours are drawn a shade greyer so their shape stays readable, animals turn their side to the viewer, small beings (a bird, a fish) are drawn at least 1.2 m big, mist thins out between the viewer and the dreamer, and snow by night is moonlit blue-grey.
- More words are understood: whales, robots, owls, queens and princesses of somewhere, travel phrases ("takes you to", "יוצא החוצה") and places said with ב or ל ("בעננים").
- The old Google service-account key files are gone from the tree. They are still in git history and on GitHub, so delete that key in the Google Cloud console.
