# Emojidetect

Advanced Algorithms project exploring AI-driven detection functionality. The repository includes Python services and components alongside JavaScript/HTML/CSS for any UI or auxiliary tooling.

## Features

- Python-based AI components:
  - `ai_api`: Flask-based service with Swagger UI integration.
  - `ai_engine`: Packaged Python module for AI/ML logic.
- Uses computer vision and audio feature libraries (OpenCV, NumPy, SciPy, librosa, python-speech-features).
- JWT support, SQLAlchemy integration, and Swagger UI for interactive API docs (if enabled).

## Project structure

- `ai_api/`
  - Python Flask API and its dependencies (`requirements.txt`, `setup.py`)
- `ai_engine/`
  - Packaged AI/ML engine (`setup.py`)
- Front-end assets (JavaScript, HTML, CSS) may exist in other folders within the repository.

Note: This outline is based on a partial scan of the repository and may not list every directory/file.

## Tech stack

- Python (primary)
  - Flask, Flask-SQLAlchemy, flask-swagger-ui
  - PyJWT/JWT, Click
  - OpenCV, NumPy, SciPy, librosa, python-speech-features
- JavaScript, HTML, CSS (for any UI or demos)

## Getting started

### Prerequisites

- Python 3.7+ recommended (based on classifiers in `ai_engine/setup.py`)
- pip / virtual environment tools
- Platform-specific system packages may be required for:
  - OpenCV (opencv-python wheels are typically sufficient)
  - Audio processing (librosa may require soundfile/backends provided by dependencies)

### Setup

```bash
# Clone the repository
git clone https://github.com/eikden/emojidetect.git
cd emojidetect

# Create and activate a virtual environment
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install API dependencies
pip install -r ai_api/requirements.txt

# (Optional) Install the AI engine package in editable mode
pip install -e ai_engine
```

## Running the API

The API is implemented with Flask inside `ai_api`. Depending on the app’s entrypoint (e.g., `app.py` or `wsgi.py`), one of the following patterns usually works:

```bash
# Option A: Flask CLI (set the app/module name accordingly)
export FLASK_APP=ai_api.app:app        # macOS/Linux
# set FLASK_APP=ai_api.app:app         # Windows (PowerShell)
flask run --port 5000 --reload

# Option B: Direct run (if there's a runnable script)
python ai_api/app.py
```

If Swagger UI is enabled via `flask-swagger-ui`, the interactive docs are commonly served at a path like `/docs` or `/swagger`. Check the API routes to confirm.

## Testing

`pytest` is listed in the requirements. From the repository root (or relevant package directory):

```bash
pytest -q
```

## Troubleshooting

- OpenCV installation issues:
  - Ensure you’re using a compatible Python version and platform wheels (`opencv-python`).
- Audio processing (librosa):
  - Some environments require additional system libraries (e.g., libsndfile via the `soundfile` dependency).
- If Flask fails to find the app:
  - Verify the module path used in `FLASK_APP` matches the actual package/file structure.

## Contributing

- Open an issue or submit a pull request with clear description and rationale.
- Follow existing code style and add tests where applicable.

## License

A license file was not detected. Consider adding a LICENSE to clarify usage and distribution terms.

## Acknowledgments

- Built for an Advanced Algorithms project.
- Thanks to the maintainers and contributors of the listed open-source libraries.
