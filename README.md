# TinyLlama + vLLM on Lambda Cloud (minimal E2E)

GPU inference runs on a **[Lambda Cloud](https://cloud.lambdalabs.com/)** instance. Your **laptop** runs **CrewAI** → **nginx** (load balancer) → the FastAPI **gateway** → vLLM over an **SSH tunnel**.

**You need:** Python 3.10+ on the laptop, **nginx** for Steps **12–14**, a Lambda account, and **Docker** if you follow **Steps 16–17** (Prometheus + Grafana). Docker is also used for optional **Jaeger** traces.

---

## Run everything in this order

Do the steps **in sequence**. Keep **earlier** long-running steps open (Lambda SSH with vLLM, tunnel) while you do **later** steps on **new terminal tabs** on your laptop.

---

### Step 1 — Create an SSH key on your laptop

Skip if you already have a key registered with Lambda.

```bash
ssh-keygen -t ed25519 -C "your_email@example.com" -f ~/.ssh/id_ed25519_lambda -N ""
cat ~/.ssh/id_ed25519_lambda.pub
```

Copy the **full** line from the `.pub` file (starts with `ssh-ed25519`). Never share the file **without** `.pub`.

---

### Step 2 — Add that key in Lambda Cloud

1. Open **https://cloud.lambdalabs.com/** and sign in.
2. Go to **SSH keys** (account or settings).
3. **Add** a key and paste the **public** line. Save.

---

### Step 3 — Launch a GPU instance in Lambda

1. Open **Instances** → **Launch instance** (or equivalent).
2. Pick a **region** and a **GPU** type (e.g. **1× A10** is plenty for TinyLlama 1.1B).
3. **Base image:** choose **Lambda Stack 24.04** or **Lambda Stack 22.04** (or the console label **Lambda Stack 2**). That image includes the NVIDIA driver, CUDA, and Python. Do **not** pick plain **Ubuntu Server** unless you plan to install the GPU stack yourself. Details: [Lambda base images](https://docs.lambda.ai/public-cloud/on-demand/#base-images).
4. If asked for a **filesystem:** choose **Don’t attach** unless you want paid persistent storage; TinyLlama is fine on the instance disk.
5. If asked for **firewall rulesets:** pick one that allows **inbound SSH (port 22)**. You do **not** need to open port **8000** to the internet if you use the SSH tunnel in Step 7.
6. Select your **SSH key** from Step 2.
7. **Launch** and wait until the instance is **running**.
8. Copy the instance **public IP** and the **ssh** command Lambda shows (user is usually `ubuntu`).

You pay while the instance runs; **terminate** it when done to stop billing.

---

### Step 4 — SSH from your laptop into the instance

```bash
ssh -i ~/.ssh/id_ed25519_lambda ubuntu@<INSTANCE_IP>
```

If this fails: confirm the instance is running, port 22 is allowed, and the key matches.

---

### Step 5 — On the instance: confirm the GPU

```bash
nvidia-smi
```

You should see an NVIDIA GPU. If you get **`command not found`** or no GPU, you likely chose a **CPU-only** SKU or the wrong image — terminate and launch again with a **GPU** type and **Lambda Stack** (Step 3). Do not try to “fix” a CPU instance with `apt install nvidia-utils` only.

---

### Step 6 — On the instance: install vLLM and start the server

Still **inside the SSH session** from Step 4. Leave this terminal open with **`vllm serve`** running in the foreground.

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install -U pip wheel
pip install "vllm==0.13.0"

vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --served-model-name texttinyllama \
  --host 0.0.0.0 \
  --port 8000
```

Equivalent from the repo (run on the **GPU** from a checkout of this project, or after copying **`scripts/vllm_engine/`**):

```bash
bash scripts/vllm_engine/baseline.sh
```

Other **engine-arg** profiles (chunked prefill, prefix caching, speculative, …) live in the same folder; flags follow the [vLLM **Engine arguments**](https://docs.vllm.ai/en/stable/configuration/engine_args/) reference.

**Gateway + Crew cannot “send” engine flags over HTTP.** vLLM reads chunked prefill, prefix cache, speculative config, etc. only at **`vllm serve`** startup. To pick an engine per **`python crew.py --technique …`** without restarting servers between arms, run **one vLLM per port** on the instance and let the gateway route by technique:

1. On the **GPU host:** `bash scripts/vllm_engine/run_engine_fleet.sh` (starts **:8000** … **:8005** with the same flags as the individual `*.sh` scripts). Optional **:8005** needs **`VLLM_SPECULATIVE_CONFIG_JSON`** set first. Several processes on one GPU may OOM for larger models — use a small model or stop processes you are not testing.
2. On the **laptop:** **`VLLM_BASE_URL=http://127.0.0.1:8000`**, **`VLLM_AUTO_ENGINE_ROUTING=1`** in **`.env`**, restart **`python gateway.py`**. **`GET /health`** lists **`backend_map_keys`** when routing is on.
3. **SSH tunnel** every forwarded port (Step 7 example below).

Wait until you see **application startup complete**. First install and first model download can take several minutes. If `pip` fails to build, run `sudo apt install -y build-essential` and retry.

**Note:** A log line like **`GET / HTTP/1.1" 404`** is normal — the API is under **`/v1/`**, not `/`.

---

### Step 7 — On your laptop: open a **second** terminal and start the SSH tunnel

Do **not** close Step 6. In a **new** tab on the **laptop**:

```bash
ssh -i ~/.ssh/id_ed25519_lambda -L 8000:127.0.0.1:8000 -N ubuntu@<INSTANCE_IP>
```

- **`-L 8000:127.0.0.1:8000`** forwards your laptop’s **`http://127.0.0.1:8000`** to vLLM on the instance.
- **`-N`** means no remote shell — this tab only holds the tunnel. Leave it open.

**Engine fleet** ( **`run_engine_fleet.sh`** + **`VLLM_AUTO_ENGINE_ROUTING=1`** ): forward **8000–8005** in one `ssh` (one line, multiple **`-L`**):

```bash
ssh -i ~/.ssh/id_ed25519_lambda \
  -L 8000:127.0.0.1:8000 -L 8001:127.0.0.1:8001 -L 8002:127.0.0.1:8002 \
  -L 8003:127.0.0.1:8003 -L 8004:127.0.0.1:8004 -L 8005:127.0.0.1:8005 \
  -N ubuntu@<INSTANCE_IP>
```

Skip **`-L …8005`** if you did not start speculative vLLM on the instance.

---

### Step 8 — On your laptop: check vLLM through the tunnel

**Third** terminal tab on the laptop:

```bash
curl -sS http://127.0.0.1:8000/v1/models
```

You should see JSON with **`"id":"texttinyllama"`**. If this fails, fix Step 6 or Step 7 before continuing.

---

### Step 9 — On your laptop: configure the project

Use any free tab on the **laptop** (Steps 6–7 must still be running). Go to your **project root** (folder containing `requirements.txt`, `gateway.py`, `crew.py`).

```bash
cd <path-to-your-repo>
cp .env.example .env
```

Edit **`.env`** with at least:

- **`VLLM_BASE_URL=http://127.0.0.1:8000`** (no trailing slash; requires Step 7).
- **`VLLM_SERVER_PROFILE=baseline`**

Keep **`GATEWAY_OPENAI_BASE`** on **`http://127.0.0.1:8780/v1`** as in **`.env.example`** if you will use nginx (Step 12). To skip nginx and point Crew at the gateway only, set **`GATEWAY_USE_LOAD_BALANCER=false`** and **`GATEWAY_OPENAI_BASE=http://127.0.0.1:8765/v1`**.

Optional for **cost metrics** in `.env`: **`LAMBDA_CLOUD_API_KEY=`** (paste key from Lambda dashboard → **API keys**), **`LAMBDA_INSTANCE_TYPE=`** (e.g. `gpu_1x_a10`, or leave empty to infer), **`LAMBDA_COST_USE_API=true`**. Or set **`LAMBDA_COST_USE_API=false`** and use **`GPU_HOURLY_COST_USD`** only. See **`.env.example`**.

Install Python dependencies **on the laptop** (not on Lambda):

```bash
pip install -r requirements.txt
```

---

### Step 10 — On your laptop: start the gateway

Keep Steps **6** and **7** running. New tab, stay in this tab until you are done:

```bash
cd <path-to-your-repo>
set -a && source .env && set +a
python gateway.py
```

Gateway listens on **`127.0.0.1:8765`**. Prometheus scrapes **`http://127.0.0.1:9101/metrics`** when the process starts (change **`GATEWAY_METRICS_PORT`** in **`.env`** if **9101** is busy and update **`monitoring/prometheus.yml`**). Optional JSONL: **`logs/gateway/`** (disable with **`GATEWAY_METRICS_LOG_DIR=-`**).

---

### Step 11 — On your laptop: quick check (gateway port **8765**)

Still with Step **10** running, new tab:

```bash
curl -sS http://127.0.0.1:8765/
curl -sS http://127.0.0.1:8765/health
curl -sS http://127.0.0.1:8765/v1/models
```

You want **`200`** on **`/health`** with **`"status":"ok"`** when **`VLLM_BASE_URL`** is correct, and **`texttinyllama`** in **`/v1/models`**.

---

### Step 12 — On your laptop: start nginx in front of the gateway

Run **after** Step **10**. Needs **`nginx`** installed (`nginx -v`). Edit **`monitoring/nginx-gateway-lb.conf`** if you add more **`python gateway.py`** backends on **8766**, etc.

```bash
cd <path-to-your-repo>/monitoring
nginx -t -p /tmp -c "$(pwd)/nginx-gateway-lb.conf"
nginx -p /tmp -c "$(pwd)/nginx-gateway-lb.conf"
```

Stop later:

```bash
nginx -s quit -p /tmp -c "<path-to-your-repo>/monitoring/nginx-gateway-lb.conf"
```

---

### Step 13 — On your laptop: quick check (load balancer **8780**)

With Steps **10** and **12** running:

```bash
curl -sS http://127.0.0.1:8780/health
curl -sS http://127.0.0.1:8780/v1/models
curl -sS "http://127.0.0.1:8780/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-Technique: baseline" \
  -d '{"model":"texttinyllama","messages":[{"role":"user","content":"Say hi in five words."}],"max_tokens":32}'
```

If **8780** refuses the connection, Step **12** did not start or the config path is wrong. If you get **502**, the gateway on **8765** is not up.

---

### Step 14 — On your laptop: run Crew

Steps **6**, **7**, **10**, **12**, and **13** must be good. **`GATEWAY_OPENAI_BASE`** should stay **`http://127.0.0.1:8780/v1`** unless you turned off the LB in **`.env`**.

```bash
cd <path-to-your-repo>
set -a && source .env && set +a
python crew.py --technique baseline
```

Crew waits on **`GET /v1/models`** (see **`CREW_VLLM_WAIT_S`** / **`CREW_VLLM_POLL_S`** in **`.env`**). **`CREW_LLM_STREAM=true`** (default) uses SSE through the gateway.

**Path:** Crew → nginx **8780** → gateway **8765** → tunnel → vLLM. Model name **`texttinyllama`** must match **`--served-model-name`**.

---

### Step 15 — What “full metrics” means (read once)

Gateway Prometheus text is at **`http://127.0.0.1:9101/metrics`** (`llm_gateway_*` with **`technique`** and **`server_profile`**). A readable summary lives on the gateway app: **`http://127.0.0.1:8765/metrics/summary`**. vLLM engine metrics are on **`http://127.0.0.1:8000/metrics`** via the tunnel. Per-request JSONL defaults to **`logs/gateway/*.jsonl`** (turn off with **`GATEWAY_METRICS_LOG_DIR=-`**). Grafana and Prometheus are Docker-only in this guide.

**Labels:** **`crew.py --technique`** sets **`X-Technique`**. Set **`VLLM_SERVER_PROFILE`** in **`.env`** to match the vLLM you are running; restart **`python gateway.py`** after you change the server.

After editing **`gateway.py`**, restart the gateway. Grafana is **Step 16**.

---

### Step 16 — Start Prometheus + Grafana (Docker)

**Requirements:** Steps **6** (vLLM), **7** (tunnel), and **10** (gateway) are running so something answers on laptop **`127.0.0.1:8000`** and **`127.0.0.1:9101`**.

From the **repo root**:

```bash
cd monitoring
docker compose up -d
```

**Prometheus** is at **http://127.0.0.1:9090** and reads **`monitoring/prometheus.yml`**. It scrapes **`host.docker.internal:9101`** (gateway from Step **10**) and **`host.docker.internal:8000`** (vLLM through the tunnel). **Grafana** is at **http://127.0.0.1:3000** with the Prometheus datasource and dashboards already provisioned.

**Linux:** Compose adds **`host.docker.internal:host-gateway`** for Prometheus. If targets fail, update Docker or run Prometheus on the host.

Open **http://127.0.0.1:9090/targets** and confirm **`gateway`** and **`vllm_tunnel`** are **UP**. If **`gateway`** is **DOWN**, Step **10** is not running or **9101** / **`prometheus.yml`** do not match. If **`vllm_tunnel`** is **DOWN**, fix Step **6** or **7** or confirm vLLM listens on **8000** on the instance.

**Fallback without Compose** (two separate containers, from **repo root**):

```bash
docker run -d --name prom -p 9090:9090 \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus:latest

docker run -d --name graf -p 3000:3000 grafana/grafana:latest
```

With the fallback you must **add the Prometheus datasource manually** in Grafana (**URL `http://host.docker.internal:9090`** on Mac/Win Docker Desktop, or your host IP on Linux) and **import** the JSON files under **`monitoring/grafana_dashboards/`** (Dashboards → Import → upload each file).

---

### Step 17 — Open Grafana and use the dashboards

1. Open **http://127.0.0.1:3000**.
2. Log in as **`admin` / `admin`** and set a new password when prompted.
3. **Dashboards** → folder **TinyLlama**. Start with **TinyLlama — GPU, runs & cost**; use the others for detail (gateway timings, technique cost, vLLM engine). If vLLM panels are empty, check **http://127.0.0.1:8000/metrics** for real metric names and adjust queries in Grafana.

**Sanity:** **http://127.0.0.1:9090/graph** → query **`llm_gateway_info`** → expect **`extended_timing="true"`** (otherwise restart gateway from current **`gateway.py`**).

---

### Step 18 — Generate traffic so the graphs move

With Grafana open (time range **Last 1 hour** or **Last 15 minutes**):

1. Send traffic with the Step **13** **`curl`** on **8780**, or run **`python crew.py --technique baseline`** (Step **14**).
2. Wait one or two **Prometheus scrape intervals** (15s in **`prometheus.yml`**).
3. Refresh **TinyLlama — GPU, runs & cost** (or wait for auto-refresh).

**Labeled histograms** (`technique`, `server_profile`) only get data after at least one request used that label pair. The first run after a restart may look sparse until you’ve exercised each combination.

**Local JSONL** (same events as metrics): default directory **`logs/gateway/`** is created on first request after startup unless **`GATEWAY_METRICS_LOG_DIR=-`**. Files: **`gateway_metrics_YYYY-MM-DD.jsonl`** (UTC day). **`logs/`** is gitignored.

---

### Step 19 — Different vLLM engine settings (A/B)

Engine flags are documented in [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/). **`crew.py --technique`** sets **`X-Technique`**; with **`VLLM_AUTO_ENGINE_ROUTING=1`** the gateway maps **`baseline`**, **`chunked_prefill`**, **`prefix_caching`**, **`chunked_prefill_and_prefix_caching`**, **`baseline_strict`**, **`speculative_decoding`**, and **`beam_search`** (same upstream as baseline) to **ports** **8000–8005** derived from **`VLLM_BASE_URL`** (see Step 6 / **`.env.example`**). Run the fleet on the GPU host, multi-port tunnel on the laptop, then e.g. **`python crew.py --technique chunked_prefill`**.

**One process at a time** (no auto routing): on the **GPU**, from repo root (or copy **`scripts/vllm_engine/`** there):

```bash
bash scripts/vllm_engine/baseline.sh
bash scripts/vllm_engine/chunked_prefill.sh
bash scripts/vllm_engine/prefix_caching.sh
bash scripts/vllm_engine/chunked_prefill_and_prefix_caching.sh
bash scripts/vllm_engine/baseline_strict.sh   # optional hard control; drop if your vLLM rejects the flags

export VLLM_SPECULATIVE_CONFIG_JSON='{"method":"eagle","model":"YOUR/DRAFT","num_speculative_tokens":3}'
bash scripts/vllm_engine/speculative_decoding.sh
```

**All profiles at once (matches auto routing ports):** **`bash scripts/vllm_engine/run_engine_fleet.sh`**.

Use **`VLLM_SERVE_PORT=8001`** (etc.) in front of any single-script line when you manage ports manually. Spec JSON must match your vLLM version — see [speculative decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/).

**Beam** is per-request, not a global serve flag here: run **`python crew.py --technique beam_search`** vs **`baseline`** against the same server.

**After each server config:** set **`VLLM_SERVER_PROFILE`** in **`.env`**, **restart `python gateway.py`**, then run Crew. Compare Grafana by **`server_profile`**.

**Guided A/B** (prints each arm, you restart vLLM + gateway between prompts):

```bash
./scripts/run_server_ab.sh sequential
```

**Parallel** (one vLLM per port; set **`VLLM_BACKEND_MAP_JSON`** in **`.env`** like **`.env.example`**, restart gateway, then):

```bash
./scripts/run_server_ab.sh parallel
```

**Label-only sweep** (same `vllm serve`, different `X-Technique` — not a server flag test):

```bash
./scripts/run_experiments.sh
```

---

### Step 20 — Monitoring troubleshooting

Targets **DOWN**: confirm Steps **6**, **7**, and **10**, then `curl -sS http://127.0.0.1:9101/metrics` and `curl -sS http://127.0.0.1:8000/metrics`. **Grafana empty**: widen the time range; in Prometheus **Graph** try **`llm_gateway_requests_total`**. **No `llm_gateway_*`**: restart **`python gateway.py`**. **9101 refused**: free the port or change **`GATEWAY_METRICS_PORT`**. **vLLM panels empty**: check **`{job="vllm_tunnel"}`** in Prometheus for renamed metrics. **Duplicate containers**: `cd monitoring && docker compose down` or `docker rm -f prom graf`.

---

## If something goes wrong

1. **SSH permission denied** — Wrong key, user, or key not added in Lambda (Steps 1–2).
2. **`nvidia-smi` not found on Lambda** — Wrong instance type or image; redo Step 3 with a **GPU** SKU and **Lambda Stack**.
3. **`libcuda` / device errors in Python on Lambda** — Same as (2); GPU not visible.
4. **`connection refused` on laptop `127.0.0.1:8000`** — Step 6 not running, or Step 7 tunnel not running, or vLLM not bound to `0.0.0.0:8000`.
5. **`connection refused` on `127.0.0.1:8765`** — Step **10** not running or wrong directory/env.
6. **`connection refused` on `127.0.0.1:8780`** — Step **12** (nginx) not running or wrong **`nginx -c`** path.
7. **`502` from `127.0.0.1:8780`** — nginx is up but nothing on **8765**; start Step **10** or fix **`upstream`** in **`nginx-gateway-lb.conf`**.
8. **Gateway `/health` not `"ok"`** — Fix **`VLLM_BASE_URL`** in **`.env`** (**`http://127.0.0.1:8000`** with the tunnel).
9. **`Failed to export traces … 4317`** — Start Jaeger (optional below) or set **`OTEL_TRACES_EXPORTER=none`** in **`.env`**.
10. **Crew LLM errors** — Run Step **13** **`curl …/v1/models`** on **8780** (or Step **11** on **8765** without nginx); confirm tunnel and vLLM.
11. **`apt full-upgrade` errors on Lambda Stack 24.04** — Known Lambda caveat; see [Lambda base images / troubleshooting](https://docs.lambda.ai/public-cloud/on-demand/#base-images).
12. **Grafana empty / Prometheus targets red** — **Step 20**; run **`docker compose`** from **`monitoring/`** and keep Steps **6–7–10** running.

---

## Optional: OpenTelemetry traces (Jaeger)

Separate from Prometheus (request spans, not histograms). Same **Docker** dependency as Step **16**.

```bash
docker run -d --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one:latest
```

In **`.env`**: **`OTEL_TRACES_EXPORTER=otlp`** and **`OTEL_EXPORTER_OTLP_*=http://127.0.0.1:4317`**. Restart **`python gateway.py`**, run **`python crew.py`**, open **http://127.0.0.1:16686** (services **`crewai`**, **`gateway`**; match **`X-Trace-Id`** from responses).

---

## Optional: run vLLM in Docker on Lambda (instead of Step 6 pip)

Use if you want a containerized vLLM. Lambda Stack often already has Docker + NVIDIA Container Toolkit; try **`docker run --gpus all ...`** first. Otherwise install the toolkit per [NVIDIA’s guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker run --gpus all --ipc=host -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  vllm/vllm-openai:latest \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --served-model-name texttinyllama \
  --host 0.0.0.0 \
  --port 8000
```

Pin a tag from [Docker Hub](https://hub.docker.com/r/vllm/vllm-openai) instead of `:latest` if you want a fixed version. Use **`docker run -d`** to keep the server running after you disconnect (see Docker docs for logs).

Then continue from **Step 7** unchanged.

---