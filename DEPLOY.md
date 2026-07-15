# Deploying to Hugging Face Spaces

The app is a Streamlit UI. Detection runs on the Roboflow + AI APIs, so the Space
needs **no GPU** and only the light `requirements.txt`.

## 1. Create the Space
1. Go to https://huggingface.co/new-space
2. Owner: your account · Space name: `construction-defect-detection`
3. SDK: **Streamlit**
4. Visibility: **Private** (recommended — see the cost note below), or Public
5. Create Space.

## 2. Push the code
The repo already has everything the Space needs (`app.py`, `src/`, `data/`,
`requirements.txt`, `README.md` with the Space front-matter, `.streamlit/`).

```bash
# add the Space as a remote (replace <user>)
git remote add space https://huggingface.co/spaces/<user>/construction-defect-detection
git push space main
```
(Or upload the files via the Space's "Files" tab.) `.env` is gitignored and will
NOT be pushed — that is correct; set the keys as secrets instead (next step).

## 3. Add the secrets
In the Space: **Settings → Variables and secrets → New secret**, add:

| Name | Value |
|------|-------|
| `ROBOFLOW_API_KEY` | your Roboflow key |
| `OPENAI_API_KEY` | your OpenAI key |
| `ROBOFLOW_WORKSPACE` | `aakashs-workspace-zqqzu` |
| `ROBOFLOW_WORKFLOW_ID` | `detect-and-classify-2` |
| `ROBOFLOW_MODEL_ID` | `training-dataset-1gvqr/2` |
| `ROBOFLOW_API_URL` | `https://serverless.roboflow.com` |

HF injects these as environment variables, which the code already reads via
`os.getenv(...)` — no code change needed. The Space rebuilds and goes live.

## 4. Use it
- In the sidebar keep **Inference Mode = "Roboflow Workflow"** (the hosted demo
  has no local YOLO weights; "Local YOLO Weights" mode will show a friendly notice).
- Upload a photo → detections, severity and the structural element appear.

## Cost / privacy note (important)
Every run spends **your** Roboflow + OpenAI credits using your keys. On a **public**
Space, any visitor can trigger that. Keep the Space **private**, or share only with
your professor, unless you intend to fund public usage.

## Alternative: Streamlit Community Cloud
1. Push the repo to GitHub.
2. https://share.streamlit.io → New app → pick the repo → main file `app.py`.
3. **Advanced settings → Secrets**, paste the same keys in TOML form:
   ```toml
   ROBOFLOW_API_KEY = "..."
   OPENAI_API_KEY = "..."
   ROBOFLOW_WORKSPACE = "aakashs-workspace-zqqzu"
   ROBOFLOW_WORKFLOW_ID = "detect-and-classify-2"
   ROBOFLOW_MODEL_ID = "training-dataset-1gvqr/2"
   ROBOFLOW_API_URL = "https://serverless.roboflow.com"
   ```
   (Free Streamlit Cloud apps are public-only.)

## Local run
```bash
pip install -r requirements-dev.txt   # full deps incl. training/reports
streamlit run app.py
```
