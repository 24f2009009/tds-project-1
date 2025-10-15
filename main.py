from pydantic import BaseModel, Field
from typing import List
import requests
import os
import base64
import time
import json
from fastapi import FastAPI, BackgroundTasks, Request
from dotenv import load_dotenv
from pydantic_ai import Agent, Tool
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

load_dotenv()

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
SECRET_KEY = os.getenv('SECRET_KEY')
GITHUB_USERNAME = "24f2009009"


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
        f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages",
        headers=headers,
        json=payload
    )
    if response.status_code not in (201, 409):  # 409 means already exists
        raise Exception(f"failed to enable github pages: {response.status_code}, {response.text}")


def wait_for_pages_enabled(repo_name: str, timeout: int = 120, interval: int = 10):
    """
    Polls GitHub Pages API until the site is ready or timeout is reached.
    """
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "accept": "application/vnd.github+json"}
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/pages"

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
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_path}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


def get_latest_commit_sha(repo_name: str, branch: str = "main"):
    """Get the SHA of the latest commit on a branch."""
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "accept": "application/vnd.github+json"
    }
    url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/commits/{branch}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json().get("sha")
    raise Exception(f"Failed to get commit SHA: {resp.status_code}, {resp.text}")


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

        url = f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{file_name}"
        response = requests.put(url, headers=headers, json=payload)

        if response.status_code not in (200, 201):
            raise Exception(
                f"Failed to push {file_name}: {response.status_code}, {response.text}"
            )


def download_attachment(attachment: dict) -> str:
    """
    Download attachment content from data URI or regular URL.
    Returns the content as a string (text or base64 for images).
    """
    url = attachment.get("url", "")
    name = attachment.get("name", "unknown")
    
    # Handle data URIs
    if url.startswith("data:"):
        # Extract the actual data from data URI
        # Format: data:mime/type;base64,<data>
        try:
            parts = url.split(",", 1)
            if len(parts) == 2:
                return parts[1]  # Return the base64 data
            return url
        except Exception as e:
            print(f"⚠️ Failed to parse data URI for {name}: {e}")
            return url
    
    # Handle GitHub URLs (convert to raw content URL)
    elif "github.com" in url:
        try:
            # Convert GitHub UI URL to raw content URL
            raw_url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
            print(f"📥 Downloading from GitHub: {raw_url}")
            resp = requests.get(raw_url, timeout=15)
            if resp.status_code == 200:
                # If it's an image or binary, return as base64
                if any(ext in name.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg']):
                    return base64.b64encode(resp.content).decode()
                # Otherwise return as text
                return resp.text
            else:
                print(f"⚠️ Failed to download from GitHub: {resp.status_code}")
                return f"[Failed to download: {raw_url}]"
        except Exception as e:
            print(f"⚠️ Error downloading from GitHub: {e}")
            return f"[Error downloading: {url}]"
    
    # Handle regular URLs
    elif url.startswith("http://") or url.startswith("https://"):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                # If it's an image or binary, return as base64
                if any(ext in name.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg']):
                    return base64.b64encode(resp.content).decode()
                # Otherwise return as text, with special characters escaped for JSON safety
                text = resp.text
                # Escape backslashes and quotes for JSON safety
                text = text.replace('\\', '\\\\').replace('"', '\\"')
                return text
            else:
                print(f"⚠️ Failed to download {url}: {resp.status_code}")
                return f"[Failed to download: {url}]"
        except Exception as e:
            print(f"⚠️ Error downloading {url}: {e}")
            return f"[Error downloading: {url}]"
    
    return url


def format_attachments_for_llm(attachments: list | None) -> str:
    """
    Format attachments for LLM consumption.
    Downloads and structures attachment data.
    """
    if not attachments:
        return "None"
    
    formatted = []
    for att in attachments:
        name = att.get("name", "unknown")
        url = att.get("url", "")
        
        print(f"📎 Processing attachment: {name} from {url}")
        
        # Download the actual content
        content = download_attachment(att)
        
        # Check if download failed
        if content.startswith("[Failed") or content.startswith("[Error"):
            formatted.append(f"File: {name}\nURL: {url}\nStatus: {content}")
            continue
        
        # Format based on file type
        if any(ext in name.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg']):
            # For images, provide data URI format
            mime_type = "image/png"  # default
            if ".jpg" in name.lower() or ".jpeg" in name.lower():
                mime_type = "image/jpeg"
            elif ".gif" in name.lower():
                mime_type = "image/gif"
            elif ".webp" in name.lower():
                mime_type = "image/webp"
            elif ".svg" in name.lower():
                mime_type = "image/svg+xml"
            
            data_uri = f"data:{mime_type};base64,{content}"
            formatted.append(f"File: {name}\nType: Image ({mime_type})\nFull Data URI: {data_uri[:200]}... (use complete data URI in your code)\nNote: Embed this image in HTML using: <img src=\"{data_uri}\" alt=\"{name}\">")
        else:
            # For text files, show full content
            formatted.append(f"File: {name}\nType: Text\nFull Content:\n{content}")
    
    return "\n\n---\n\n".join(formatted)


def write_code_with_llm(
    brief: str,
    description: str = "",
    attachments: list | None = None,
    checks: list | None = None,
) -> list[dict]:
    """
    Generates a single-page web app using a pydantic-ai agent.
    Allows the agent to use DuckDuckGo for contextual search.
    """

    system_prompt = (
        "You are a professional web developer. "
        "Generate a production-ready single-page HTML app with inline CSS/JS only. "
        "CRITICAL: ALL content must be embedded DIRECTLY in the HTML file. "
        "DO NOT use fetch(), XMLHttpRequest, or any file loading - embed content as JavaScript variables. "
        "When attachments are provided, their FULL CONTENT is included in the prompt. "
        "For text attachments: Embed the content as a JavaScript template literal (backticks) or string. "
        "For image attachments: Embed using complete data URIs in <img> tags or JavaScript. "
        "DO NOT use placeholder content - use the ACTUAL attachment content provided. "
        "For README.md: Use actual newlines (\\n), not the literal text '\\n'. "
        "Format README properly with real line breaks, not escaped characters."
    )

    checks_text = "\n".join(f"- {c}" for c in checks or []) if isinstance(checks, list) else "None"
    
    # Format attachments with actual content
    attachments_text = format_attachments_for_llm(attachments)

    user_prompt = f"""
User task:
{brief}

Checks to satisfy (these are JavaScript checks that will be run on the page):
{checks_text}

Existing context:
{description or "(none)"}

Attachments (FULL CONTENT DOWNLOADED AND PROVIDED BELOW):
{attachments_text}

CRITICAL REQUIREMENTS:
1. ALL files must be self-contained - NO external file loading (no fetch, no XMLHttpRequest)
2. Embed attachment content DIRECTLY in JavaScript as variables/constants
3. For text files: Use JavaScript template literals like: const content = `actual content here`;
4. For images: Use data URIs directly in HTML: <img src="data:image/...;base64,..." />
5. README.md must have REAL newlines, not literal \\n characters
6. Ensure all checks pass - they test for specific elements/functionality
7. Load external libraries (marked, highlight.js) from CDN via <script> tags

EXAMPLE for markdown file:
```javascript
const markdownContent = `# Your Title
This is the actual content from the attachment.
Code blocks work too.`;
// Then use: marked.parse(markdownContent)
```

Output format (valid JSON only):
{{ "files": [ 
    {{ "name": "index.html", "content": "<!DOCTYPE html>..." }},
    {{ "name": "README.md", "content": "# Title\\n\\n## Summary\\n..." }} 
] }}

Remember: Use the EXACT content from attachments above, embedded directly in your code.
"""

    # Define the search tool with typed argument
    def duckduckgo_search(query: str) -> str:
        """Search the web using DuckDuckGo and return summarized results."""
        return duckduckgo_search_tool(query)

    search_tool = Tool(
        name="duckduckgo_search",
        description="Search DuckDuckGo for helpful web results.",
        function=duckduckgo_search,
    )

    agent = Agent(
        model="openai:gpt-5-nano",
        system_prompt=system_prompt,
        tools=[search_tool],
    )

    result = agent.run_sync(user_prompt)

    try:
        # First try to parse the JSON as-is
        try:
            data = json.loads(result.output)
        except json.JSONDecodeError as je:
            # If parsing fails, try to clean/escape the output
            cleaned_output = result.output.replace('\\', '\\\\').replace('\n', '\\n')
            try:
                data = json.loads(cleaned_output)
            except json.JSONDecodeError:
                raise Exception(f"Failed to parse LLM output even after cleaning: {je}")
        
        # Validate the structure
        validated = LLMResponse(**data)
    except Exception as e:
        raise Exception(f"Failed to parse or validate LLM output: {e}")

    return [f.model_dump() for f in validated.files]


def send_evaluation_callback(data: dict, repo_name: str):
    """Send evaluation result to callback URL."""
    headers = {"Content-Type": "application/json"}
    evaluation_url = data["evaluation_url"]

    # Get the actual commit SHA (not file SHA)
    commit_sha = get_latest_commit_sha(repo_name)
    repo_url = f"https://github.com/{GITHUB_USERNAME}/{repo_name}"
    pages_url = f"https://{GITHUB_USERNAME}.github.io/{repo_name}/"

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
    try:
        files = write_code_with_llm(
            brief=data["brief"],
            description=data.get("description", ""),
            attachments=data.get("attachments"),
            checks=data.get("checks", [])
        )
        repo_name = f"{data['task']}"
        create_github_repo(repo_name)
        push_files_to_repo(repo_name, files, 1)
        enable_github_pages(repo_name)
        wait_for_pages_enabled(repo_name)
        send_evaluation_callback(data, repo_name)
    except Exception as e:
        print(f"❌ Round 1 failed: {e}")


def round2(data):
    try:
        repo_name = f"{data['task']}"
        headers = {"Authorization": f"Bearer {GITHUB_TOKEN}", "accept": "application/vnd.github+json"}

        # Fetch both index.html and README.md
        files_content = {}
        for filename in ["index.html", "README.md"]:
            response = requests.get(
                f"https://api.github.com/repos/{GITHUB_USERNAME}/{repo_name}/contents/{filename}",
                headers=headers
            )
            if response.status_code != 200:
                print(f"Warning: Failed to fetch {filename}: {response.status_code}, {response.text}")
                continue
                
            files_content[filename] = base64.b64decode(response.json()["content"]).decode("utf-8")

        # Prepare existing content description
        existing_description = []
        if "index.html" in files_content:
            existing_description.append(f"Existing index.html (truncated if too long):\n{files_content['index.html'][:4000]}")
        if "README.md" in files_content:
            existing_description.append(f"Existing README.md:\n{files_content['README.md']}")
        
        description = "\n\n---\n\n".join(existing_description)

        updated_files = write_code_with_llm(
            brief=f"Refactor or extend the following existing files according to the new brief. Maintain consistency between index.html and README.md:\n{data['brief']}",
            description=description,
            attachments=data.get("attachments"),
            checks=data.get("checks", [])
        )

        push_files_to_repo(repo_name, updated_files, 2)
        wait_for_pages_enabled(repo_name)
        send_evaluation_callback(data, repo_name)
    except Exception as e:
        print(f"❌ Round 2 failed: {e}")


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
    uvicorn.run(app, host="0.0.0.0", port=7860)