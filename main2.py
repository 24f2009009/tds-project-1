from pydantic import BaseModel, Field
from typing import List
import requests
import os
import base64
import time
import json
from fastapi import FastAPI, BackgroundTasks, Request
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
SECRET_KEY = os.getenv('SECRET_KEY')


# ----------------------------- Models -----------------------------
class GeneratedFile(BaseModel):
    name: str = Field(..., description="Filename including extension")
    content: str = Field(..., description="File content as plain text")


class LLMResponse(BaseModel):
    files: List[GeneratedFile]


# ----------------------------- Helper Functions -----------------------------
def validate_secret(secret: str) -> bool:
    return secret == SECRET_KEY


def create_github_repo(repo_name: str):
    payload = {"name": repo_name, "private": False, "auto_init": False, "license_template": "mit"}
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "accept": "application/vnd.github+json"}
    response = requests.post("https://api.github.com/user/repos", headers=headers, json=payload)
    if response.status_code != 201:
        raise Exception(f"failed to create repo: {response.status_code}, {response.text}")
    return response.json()


def enable_github_pages(repo_name: str):
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "accept": "application/vnd.github+json"}
    payload = {"build_type": "legacy", "source": {"branch": "main", "path": "/"}}
    response = requests.post(
        f"https://api.github.com/repos/24f2009009/{repo_name}/pages",
        headers=headers,
        json=payload
    )
    if response.status_code != 201:
        raise Exception(f"failed to enable github pages: {response.status_code}, {response.text}")


def wait_for_pages_enabled(repo_name: str, timeout: int = 120, interval: int = 10):
    """
    Polls GitHub Pages API until the site is ready or timeout is reached.
    """
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/24f2009009/{repo_name}/pages"

    start = time.time()
    while True:
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            status = resp.json().get("status", "")
            if status in ("built", "ready"):
                print("✅ GitHub Pages is live.")
                return
            else:
                print(f"⏳ Waiting for GitHub Pages... (status: {status})")
        else:
            print(f"⚠️ Pages status check failed: {resp.status_code}")

        if time.time() - start > timeout:
            raise TimeoutError("Timed out waiting for GitHub Pages to become ready.")
        time.sleep(interval)


def get_file_sha(repo_name: str, file_path: str):
    """Check if a file already exists and return its SHA if present."""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "accept": "application/vnd.github+json"
    }
    url = f"https://api.github.com/repos/24f2009009/{repo_name}/contents/{file_path}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


def push_files_to_repo(repo_name: str, files: dict, round_num: int):
    """Push multiple files to a repo (creates or updates cleanly)."""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "accept": "application/vnd.github+json"
    }

    for file in files:
        file_name = file["name"]
        file_content = file["content"]

        encoded_content = base64.b64encode(file_content.encode()).decode()
        sha = get_file_sha(repo_name, file_name)
        payload = {
            "message": f"Round {round_num}: update {file_name}",
            "content": encoded_content,
        }
        if sha:
            payload["sha"] = sha

        url = f"https://api.github.com/repos/24f2009009/{repo_name}/contents/{file_name}"
        response = requests.put(url, headers=headers, json=payload)

        if response.status_code not in (200, 201):
            raise Exception(
                f"Failed to push {file_name}: {response.status_code}, {response.text}"
            )




def write_code_with_llm(
    brief: str,
    description: str = "",
    attachments: list | None = None,
    checks: list | None = None
) -> list[dict]:
    """
    Uses OpenAI API (via aipipe.org endpoint) to generate a single-page web app.
    Supports attachments by fetching their content. All CSS/JS is inlined.
    Validates output against LLMResponse schema.
    """

    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://aipipe.org/openai/v1')

    # Prepare checks text
    checks_text = "\n".join(f"- {c}" for c in checks or []) if isinstance(checks, list) else "None"

    # Fetch attachment contents
    attachment_texts = []
    for url in attachments or []:
        try:
            r = requests.get(url, timeout=5)
            r.raise_for_status()
            content_preview = r.text[:2000]  # truncate if too long
            attachment_texts.append(f"URL: {url}\nContent (truncated to 2000 chars):\n{content_preview}")
        except Exception as e:
            attachment_texts.append(f"URL: {url}\nCould not fetch content: {e}")

    attachments_text = "\n\n".join(attachment_texts) if attachment_texts else "None"

    # Separate system prompt and user prompt
    system_prompt = (
        "You are a professional web developer. "
        "Generate clean, production-ready HTML for a single-page web app."
    )

    user_prompt = f"""
User task:

{brief}

Checks to satisfy:
{checks_text}

Important Requirements:
- Generate **at least** these files:
  - index.html: complete single-page web app with inline CSS/JS.
  - README.md: a markdown file describing the project, usage instructions, and any relevant notes.
- Do not reference external `.css` or `.js` files.
- May use CDNs for libraries if needed.
- Visually polished, modern, and responsive design.
- Use descriptive element IDs as specified in the brief.
- Valid and accessible HTML.
- Avoid placeholders like `${{seed}}`.

Attachments content (if any):
{attachments_text}

Existing code or description:
{description or "(none)"}

Output only the files in JSON format exactly matching this schema:
{{ "files": [ {{ "name": string, "content": string }} ] }}
"""


    # Call OpenAI API
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "gpt-5-nano",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"},
    }

    resp = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload)
    if resp.status_code != 200:
        raise Exception(f"OpenAI request failed: {resp.status_code}, {resp.text}")

    try:
        content = resp.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        validated = LLMResponse(**data)
    except Exception as e:
        raise Exception(f"Failed to parse or validate LLM output: {e}")

    return [f.model_dump() for f in validated.files]




def send_evaluation_callback(data: dict, repo_name: str):
    """Send evaluation result to callback URL."""
    headers = {"Content-Type": "application/json"}
    evaluation_url = data["evaluation_url"]

    commit_sha = get_file_sha(repo_name,'index.html')
    repo_url = f"https://github.com/24f2009009/{repo_name}"
    pages_url = f"https://24f2009009.github.io/{repo_name}/"

    payload = {
        "email": data["email"],
        "task": data["task"],
        "round": data["round"],
        "nonce": data["nonce"],
        "repo_url": repo_url,
        "commit_sha": commit_sha,
        "pages_url": pages_url,
    }

    delay = 1
    for _ in range(5):
        try:
            resp = requests.post(evaluation_url, json=payload, headers=headers)
            if resp.status_code == 200:
                print("✅ Evaluation callback sent successfully.")
                return
            else:
                print(f"⚠️ Callback failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            print(f"Error sending callback: {e}")
        time.sleep(delay)
        delay *= 2


# ----------------------------- Rounds -----------------------------
def round1(data):
    files = write_code_with_llm(
        brief=data["brief"],
        description=data.get("description", ""),
        attachments=data.get("attachments"),
        checks=data.get("checks", [])
    )
    repo_name = f"{data['task']}_{data['nonce']}"
    create_github_repo(repo_name)
    push_files_to_repo(repo_name, files, 1)
    enable_github_pages(repo_name)
    wait_for_pages_enabled(repo_name)
    send_evaluation_callback(data, repo_name)


def round2(data):
    repo_name = f"{data['task']}_{data['nonce']}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "accept": "application/vnd.github+json"}

    response = requests.get(
        f"https://api.github.com/repos/24f2009009/{repo_name}/contents/index.html",
        headers=headers
    )
    if response.status_code != 200:
        raise Exception(f"failed to fetch existing file: {response.status_code}, {response.text}")

    current_content = base64.b64decode(response.json()["content"]).decode("utf-8")
    existing_snippet = current_content[:4000]

    updated_files = write_code_with_llm(
        brief=f"Refactor or extend the following existing page according to the new brief:\n{data['brief']}",
        description=f"Existing code (truncated if too long):\n{existing_snippet}",
        attachments=data.get("attachments"),
        checks=data.get("checks", [])
    )

    push_files_to_repo(repo_name, updated_files, 2)
    wait_for_pages_enabled(repo_name)
    send_evaluation_callback(data, repo_name)


# ----------------------------- FastAPI App -----------------------------
app = FastAPI()

@app.post("/handle_task")
async def handle_task(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()

    if not validate_secret(data.get("secret", "")):
        return {"error": "Invalid secret"}

    round_num = data.get("round")
    if round_num == 1:
        background_tasks.add_task(round1, data)
        return {"message": "Round 1 started in background"}
    elif round_num == 2:
        background_tasks.add_task(round2, data)
        return {"message": "Round 2 started in background"}
    else:
        return {"error": "Invalid round"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
