from flask import Flask, render_template, Response
import subprocess
import time
import requests

app = Flask(__name__)

GRADIO_URL = "http://localhost:7860"  # Replace with your Gradio app's URL if different
NGROK_PORT = 7860  # Port your Gradio app is running on

# Function to start the Gradio app in the background
def start_gradio_app():
    print("Starting Gradio app...")
    process = subprocess.Popen(["python", "app_VTON.py", "--load_mode", "8bit"])
    time.sleep(10)  # Give Gradio some time to start
    print("Gradio app started (hopefully!).")
    return process

# Function to start ngrok
def start_ngrok():
    print("Starting ngrok...")
    process = subprocess.Popen(["ngrok", "http", "--authtoken", "YOUR_AUTH_TOKEN", str(NGROK_PORT)])
    time.sleep(5)
    print("ngrok started.")
    return process

# Function to get the public ngrok URL
def get_ngrok_url():
    try:
        res = requests.get("http://localhost:4040/api/tunnels")
        res_json = res.json()
        for tunnel in res_json["tunnels"]:
            if tunnel["proto"] == "https":
                return tunnel["public_url"]
        return None
    except requests.exceptions.ConnectionError:
        return None

gradio_process = None
ngrok_process = None
ngrok_url = None

@app.route("/")
def index():
    global gradio_process, ngrok_process, ngrok_url
    if gradio_process is None or gradio_process.poll() is not None:
        gradio_process = start_gradio_app()
    if ngrok_process is None or ngrok_process.poll() is not None:
        ngrok_process = start_ngrok()
        ngrok_url = get_ngrok_url()

    return render_template('index.html', ngrok_url=ngrok_url)

# Simple proxy to forward requests to the Gradio app (optional, but helpful)
@app.route("/gradio/<path:path>")
def gradio_proxy(path):
    if ngrok_url:
        target_url = f"{GRADIO_URL}/{path}"
        try:
            resp = requests.get(target_url, stream=True)
            return Response(resp.iter_content(chunk_size=10*1024),
                            content_type=resp.headers['content-type'])
        except requests.exceptions.ConnectionError:
            return "Gradio app is not yet available.", 503
    else:
        return "Ngrok tunnel not yet established.", 503

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
