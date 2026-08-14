# Vision Ai - Image Recognition 

This turns your MobileNetV2 image classifier into a two-screen web app:

- **Home page** — a modern landing screen (hero title, feature cards, "how it
  works" pipeline) with a **Launch Classifier** button.
- **Classifier page** — the actual working tool: upload a photo (or use the
  built-in sample), click **Classify**, and see the top-5 predictions as
  animated confidence bars plus a data table.

---

## 1. Install dependencies

In your project folder (same one where `app.py` lives):

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

(This installs Streamlit, TensorFlow, Pillow, NumPy, and Pandas.)

---

## 2. Run it locally

```bash
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`. If it doesn't,
open that URL manually.

You'll land on the **home page** first. Click **🚀 Launch Classifier** to go
to the working tool. Use **← Home** in the top-right to go back.

---

## 3. Using the classifier

1. Either drag-and-drop / browse for a photo, **or** click
   **"🐒 Or try the sample image instead"**.
2. Click **✨ Classify this image**.
3. Results appear on the right: the top guess in large text, a confidence
   bar for each of the top 5 predictions, and an expandable data table.

The model is loaded once and cached (`@st.cache_resource`), so after the
first prediction, every subsequent one is fast — no reloading.

---

## 4. Deploying it so others can use it (no local setup needed for them)

### Option A — Streamlit Community Cloud (free, easiest)
1. Push your project (`app.py`, `requirements.txt`) to a public GitHub repo.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, pick your repo/branch, set the main file to `app.py`.
4. Click **Deploy**. You'll get a public URL like
   `https://your-app-name.streamlit.app`.

> Note: TensorFlow is a large dependency; Community Cloud's free tier has
> limited RAM. If the app fails to start due to memory, consider swapping
> `tensorflow` for `tensorflow-cpu` in `requirements.txt` (smaller install)
> or use a lighter model.

### Option B — Hugging Face Spaces (free, generous resources)
1. Create a new Space at https://huggingface.co/new-space, SDK = **Streamlit**.
2. Upload `app.py` and `requirements.txt` (or connect a git repo).
3. The Space builds and hosts automatically — you get a public URL.

### Option C — Your own server / Docker
```bash
# Minimal Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build and run:
```bash
docker build -t image-recognition-app .
docker run -p 8501:8501 image-recognition-app
```

---

## 5. Customizing the look

All styling lives in the `CUSTOM_CSS` string near the top of `app.py`:
- Change the gradient colors in `.hero-title` / `.top-pred-label` to rebrand.
- Change `.stApp { background: ... }` for a different overall theme.
- Swap the emoji icons in the feature cards for anything you like.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'streamlit'` | Activate your venv, then `pip install -r requirements.txt`. |
| App loads but model download hangs | Check internet connection; first run downloads ~14 MB of weights, cached afterward in `~/.keras/`. |
| Page looks unstyled / plain | Hard-refresh the browser (Cmd/Ctrl+Shift+R) — sometimes cached CSS lingers. |
| Deployed app crashes with "out of memory" | Use `tensorflow-cpu` instead of `tensorflow` in `requirements.txt`, or deploy on a host with more RAM (Hugging Face Spaces works well for this). |
