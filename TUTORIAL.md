# FastAPI + Python + 

- Activate virtual env

```bash
python -m venv venv

.\venv\Scripts\Activate.ps1
```

- pip installations

```bash
pip install fastapi uvicorn

pip freeze > requirements.txt
```

- The code is in `main.py`

### To run this app?

```bash
uvicorn main:app --reload
```

It will open up on: `http://127.0.0.1:8000/`

## The Magic of FastAPI

Visit: ```http://127.0.0.1:8000/docs```
It has:
    - User Interface By FastAPI
    - All your routes
    - Schemas
    - Could be edited