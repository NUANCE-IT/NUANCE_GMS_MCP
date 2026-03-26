# Installation

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.12 |
| RAM | 8 GB | 16 GB |
| GPU (for Ollama) | None (CPU OK) | NVIDIA RTX 3080+ |
| GMS version | 3.60 | 3.62 |
| OS | Windows 10 | Windows 11 |

---

## Scenario A — Simulation mode (no microscope)

Ideal for development, testing, and demonstrations on any workstation.

```bash
pip install "gms-mcp[ollama]"

# Pull a model
ollama pull qwen2.5:7b

# Launch interactive session
GMS_SIMULATE=1 python -m gms_mcp.client
```

---

## Scenario B — Live GMS connection

### Step 1: Install on the microscope PC (inside GMS environment)

```bat
:: Activate the GMS virtual environment
cd C:\ProgramData\Miniconda3\envs\GMS_VENV_PYTHON
activate GMS_VENV_PYTHON

:: Install ZeroMQ bridge dependency
pip install pyzmq

:: Copy the DM plugin to a convenient location
copy src\gms_mcp\dm_plugin.py C:\GMS_Scripts\dm_plugin.py
```

### Step 2: Start the bridge inside GMS

Open the GMS Python console and run:

```python
exec(open("C:/GMS_Scripts/dm_plugin.py").read())
# Output: [GMS-MCP] DM bridge ready on tcp://0.0.0.0:5555
```

### Step 3: Install GMS-MCP on your workstation

```bash
pip install "gms-mcp[ollama,zmq]"
```

### Step 4: Connect

```bash
# Point the MCP server at the live GMS bridge
GMS_MCP_ZMQ=tcp://microscope-pc:5555 python -m gms_mcp.client
```

---

## Scenario C — Remote access via Claude.ai

```bash
# Start the HTTP server on the microscope PC (or any networked machine)
python -m gms_mcp.server --transport http --port 8000

# Expose via HTTPS (for Claude.ai; use ngrok for quick testing)
ngrok http 8000
```

Then in Claude.ai → **Settings → Connectors → Add custom connector**
and enter the HTTPS URL shown by ngrok.

---

## Installing from source

```bash
git clone https://github.com/NUANCE-IT/Gatan_MCP
cd Gatan_MCP
pip install -e ".[all]"

# Verify
GMS_SIMULATE=1 pytest tests/ -v -m "not ollama"
```

---

## Troubleshooting

**`ImportError: No module named 'DigitalMicrograph'`**
This is expected outside GMS. The DMSimulator activates automatically.
Set `GMS_SIMULATE=1` to force simulation mode explicitly.

**`ConnectionRefusedError` on ZeroMQ port 5555**
The DM bridge is not running. Open the GMS Python console and re-run
`exec(open("dm_plugin.py").read())`.

**Ollama model not found**
Run `ollama list` to see pulled models. Pull with `ollama pull qwen2.5:7b`.

**Slow tool-calling with llama3.2:3b**
Switch to `qwen2.5:7b` — it achieves significantly better tool-calling
accuracy at similar latency.
