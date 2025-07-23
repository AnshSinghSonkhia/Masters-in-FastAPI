# 🚀 Masters in FastAPI

> A comprehensive guide to master FastAPI for interviews and production applications

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/) [![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
---

## 📚 Table of Contents

- [Introduction](#-introduction)
- [Core Concepts](#-core-concepts)
- [Request Handling](#-request-handling)
- [Response Models](#-response-models)
- [Dependency Injection](#-dependency-injection)
- [Database Integration](#-database-integration)
- [Authentication & Security](#-authentication--security)
- [Background Tasks & WebSockets](#-background-tasks--websockets)
- [Testing](#-testing)
- [Performance & Optimization](#-performance--optimization)
- [Deployment](#-deployment)
- [Advanced Features](#-advanced-features)
- [Common Interview Questions](#-common-interview-questions)
- [Best Practices](#-best-practices)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Introduction

### What is FastAPI?
- **Modern Python Web Framework** - Built on Starlette and Pydantic
- **High Performance** - On par with NodeJS and Go
- **Easy to Use** - Intuitive and pythonic
- **Standards-based** - OpenAPI, JSON Schema, OAuth2
- **Type Hints** - Full editor support with autocompletion

### Why FastAPI?
- ✅ **Fast Development** - Less code, fewer bugs
- ✅ **Fast Performance** - Starlette + Pydantic + Python 3.7+
- ✅ **Auto Documentation** - Swagger UI & ReDoc
- ✅ **Type Safety** - Python type hints validation
- ✅ **Async Support** - Built-in asynchronous capabilities
- ✅ **Production Ready** - Used by Microsoft, Uber, Netflix
- ✅ **Great DX** - Editor support, debugging, less time reading docs

### FastAPI vs Others
```python
# FastAPI vs Flask
- FastAPI: Async, automatic validation, auto docs
- Flask: Mature, simple, large ecosystem

# FastAPI vs Django
- FastAPI: API-focused, modern, fast
- Django: Full-stack, batteries included, admin panel

# FastAPI vs Express.js
- FastAPI: Type safety, auto validation, Python
- Express: JavaScript, minimal, flexible
```

---

## 🎯 Core Concepts

### Basic Application Structure

#### Minimal App
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Run with: uvicorn main:app --reload
```

#### With Async
```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "Async World"}
```

### Path Operations

#### HTTP Methods
```python
@app.get("/items/")         # GET - Retrieve data
@app.post("/items/")        # POST - Create data
@app.put("/items/{id}")     # PUT - Update data
@app.patch("/items/{id}")   # PATCH - Partial update
@app.delete("/items/{id}")  # DELETE - Remove data
```

#### Path Parameters
```python
@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}

# Enum path parameters
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    return {"model": model_name}
```

#### Query Parameters
```python
# Optional parameters with defaults
@app.get("/items/")
async def read_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}

# Required query parameters
@app.get("/items/search")
async def search_items(q: str):
    return {"query": q}

# Optional with None
from typing import Optional

@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
```

### Request Body

#### Pydantic Models
```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Item(BaseModel):
    name: str
    price: float
    description: Optional[str] = None
    tax: Optional[float] = None
    tags: list[str] = []
    created_at: datetime = datetime.now()

@app.post("/items/")
async def create_item(item: Item):
    return {"item": item}
```

#### Nested Models
```python
class Image(BaseModel):
    url: str
    name: str

class Product(BaseModel):
    name: str
    price: float
    images: list[Image] = []

@app.post("/products/")
async def create_product(product: Product):
    return product
```

---

## 📥 Request Handling

### Multiple Parameters

#### Combined Parameters
```python
@app.put("/items/{item_id}")
async def update_item(
    item_id: int,                    # Path parameter
    item: Item,                      # Request body
    q: Optional[str] = None         # Query parameter
):
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result
```

### Form Data
```python
from fastapi import Form

@app.post("/login/")
async def login(
    username: str = Form(...), 
    password: str = Form(...)
):
    return {"username": username}
```

### File Uploads
```python
from fastapi import File, UploadFile

# Single file
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(contents)
    }

# Multiple files
@app.post("/uploadfiles/")
async def create_upload_files(
    files: list[UploadFile] = File(...)
):
    return {"filenames": [file.filename for file in files]}
```

### Request Headers
```python
from fastapi import Header
from typing import Optional

@app.get("/items/")
async def read_items(
    user_agent: Optional[str] = Header(None),
    x_token: Optional[str] = Header(None)
):
    return {
        "User-Agent": user_agent,
        "X-Token": x_token
    }
```

### Cookies
```python
from fastapi import Cookie
from typing import Optional

@app.get("/items/")
async def read_items(
    session_id: Optional[str] = Cookie(None)
):
    return {"session_id": session_id}
```

---

## 📤 Response Models

### Response Model
```python
class UserIn(BaseModel):
    username: str
    password: str
    email: str
    full_name: Optional[str] = None

class UserOut(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None

@app.post("/users/", response_model=UserOut)
async def create_user(user: UserIn):
    # Password not included in response
    return user
```

### Response Status Codes
```python
from fastapi import status

@app.post("/items/", status_code=status.HTTP_201_CREATED)
async def create_item(item: Item):
    return item

# Custom status code
@app.delete("/items/{item_id}", status_code=204)
async def delete_item(item_id: int):
    return None
```

### Multiple Response Models
```python
from typing import Union

class ItemBasic(BaseModel):
    name: str
    price: float

class ItemDetailed(BaseModel):
    name: str
    price: float
    description: str
    tax: float

@app.get("/items/{item_id}", response_model=Union[ItemBasic, ItemDetailed])
async def read_item(item_id: int, detailed: bool = False):
    if detailed:
        return ItemDetailed(...)
    return ItemBasic(...)
```

### Response Headers
```python
from fastapi import Response

@app.get("/legacy/")
async def get_legacy_data(response: Response):
    response.headers["X-Cat-Dog"] = "alone in the world"
    return {"message": "Hello World"}
```

---

## 💉 Dependency Injection

### Basic Dependencies
```python
from fastapi import Depends

async def common_parameters(
    q: Optional[str] = None, 
    skip: int = 0, 
    limit: int = 100
):
    return {"q": q, "skip": skip, "limit": limit}

@app.get("/items/")
async def read_items(commons: dict = Depends(common_parameters)):
    return commons
```

### Class Dependencies
```python
class CommonQueryParams:
    def __init__(self, q: Optional[str] = None, skip: int = 0, limit: int = 100):
        self.q = q
        self.skip = skip
        self.limit = limit

@app.get("/items/")
async def read_items(commons: CommonQueryParams = Depends()):
    return commons
```

### Sub-dependencies
```python
def query_extractor(q: Optional[str] = None):
    return q

def query_or_default(q: str = Depends(query_extractor)):
    if not q:
        return "default"
    return q

@app.get("/items/")
async def read_items(query: str = Depends(query_or_default)):
    return {"query": query}
```

### Dependency with Yield
```python
async def get_db():
    db = DBSession()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/")
async def read_users(db: Session = Depends(get_db)):
    return db.query(User).all()
```

### Global Dependencies
```python
async def verify_token(x_token: str = Header(...)):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")

# Apply to entire app
app = FastAPI(dependencies=[Depends(verify_token)])

# Or to a router
from fastapi import APIRouter

router = APIRouter(dependencies=[Depends(verify_token)])
```

---

## 🗄️ Database Integration

### SQLAlchemy Setup

#### Database Configuration
```python
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQLALCHEMY_DATABASE_URL = "postgresql://user:password@localhost/dbname"
# For SQLite: "sqlite:///./test.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
```

#### Models
```python
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    
    items = relationship("Item", back_populates="owner")

class Item(Base):
    __tablename__ = "items"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String, index=True)
    price = Column(Float)
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    owner = relationship("User", back_populates="items")
```

#### Pydantic Schemas
```python
from pydantic import BaseModel
from typing import Optional

# User schemas
class UserBase(BaseModel):
    email: str

class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool
    items: list[Item] = []

    class Config:
        orm_mode = True

# Item schemas
class ItemBase(BaseModel):
    title: str
    description: Optional[str] = None
    price: float

class ItemCreate(ItemBase):
    pass

class Item(ItemBase):
    id: int
    owner_id: int

    class Config:
        orm_mode = True
```

#### CRUD Operations
```python
from sqlalchemy.orm import Session

# User CRUD
def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, user: UserCreate):
    fake_hashed_password = user.password + "notreallyhashed"
    db_user = User(email=user.email, hashed_password=fake_hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

# Item CRUD
def create_user_item(db: Session, item: ItemCreate, user_id: int):
    db_item = Item(**item.dict(), owner_id=user_id)
    db.add(db_item)
    db.commit()
    db.refresh(db_item)
    return db_item
```

### Async Database (SQLAlchemy 2.0)
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# Async engine
engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Async dependency
async def get_async_db():
    async with async_session() as session:
        yield session

# Async query
@app.get("/users/{user_id}")
async def read_user(user_id: int, db: AsyncSession = Depends(get_async_db)):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

### MongoDB Integration
```python
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import Document, init_beanie

# MongoDB client
client = AsyncIOMotorClient("mongodb://localhost:27017")
database = client.mydatabase

# Beanie Document
class User(Document):
    email: str
    name: str
    is_active: bool = True

    class Settings:
        collection = "users"

# Initialize Beanie
@app.on_event("startup")
async def startup_event():
    await init_beanie(database=database, document_models=[User])

# CRUD with Beanie
@app.post("/users/")
async def create_user(user: User):
    await user.insert()
    return user

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=404)
    return user
```

---

## 🔐 Authentication & Security

### Password Hashing
```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)
```

### JWT Token Authentication
```python
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

SECRET_KEY = "your-secret-key-here"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(username=username)
    if user is None:
        raise credentials_exception
    return user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
```

### OAuth2 with Scopes
```python
from fastapi.security import OAuth2PasswordBearer, SecurityScopes

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={"read": "Read access", "write": "Write access"}
)

async def get_current_user(
    security_scopes: SecurityScopes, 
    token: str = Depends(oauth2_scheme)
):
    # Verify token and check scopes
    authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    # ... token validation logic
    
    for scope in security_scopes.scopes:
        if scope not in token_scopes:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )
    return user

@app.get("/items/", dependencies=[Security(get_current_user, scopes=["read"])])
async def read_items():
    return [{"item": "Foo"}, {"item": "Bar"}]
```

### API Key Authentication
```python
from fastapi.security import APIKeyHeader, APIKeyQuery, HTTPBearer
from fastapi import Security

# Header-based API Key
api_key_header = APIKeyHeader(name="X-API-Key")

@app.get("/protected")
async def protected_route(api_key: str = Security(api_key_header)):
    if api_key != "secret-api-key":
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return {"message": "Access granted"}

# Query-based API Key
api_key_query = APIKeyQuery(name="api_key")

@app.get("/query-protected")
async def query_protected(api_key: str = Security(api_key_query)):
    # Validate API key
    return {"message": "Access granted"}
```

### CORS Configuration
```python
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:3000",
    "https://mydomain.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🔄 Background Tasks & WebSockets

### Background Tasks
```python
from fastapi import BackgroundTasks
import time

def write_notification(email: str, message=""):
    time.sleep(10)  # Simulate slow operation
    with open("log.txt", mode="a") as file:
        file.write(f"notification sent to {email}: {message}\n")

@app.post("/send-notification/{email}")
async def send_notification(
    email: str, 
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(write_notification, email, message="some notification")
    return {"message": "Notification sent in the background"}
```

### Celery Integration
```python
from celery import Celery

celery_app = Celery(
    "worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery_app.task
def process_data(data: dict):
    # Long running task
    time.sleep(60)
    return {"status": "completed", "data": data}

@app.post("/process/")
async def process_endpoint(data: dict):
    task = process_data.delay(data)
    return {"task_id": task.id}

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = process_data.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result
    }
```

### WebSockets
```python
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
            await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")
```

---

## 🧪 Testing

### Basic Testing
```python
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}

def test_create_item():
    response = client.post(
        "/items/",
        json={"name": "Test Item", "price": 10.5}
    )
    assert response.status_code == 200
    assert response.json()["name"] == "Test Item"
```

### Async Testing
```python
import pytest
from httpx import AsyncClient

@pytest.mark.anyio
async def test_async_endpoint():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
```

### Database Testing
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Test database
TEST_SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(TEST_SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Run tests
def test_create_user():
    response = client.post("/users/", json={"email": "test@example.com", "password": "secret"})
    assert response.status_code == 200
    assert response.json()["email"] == "test@example.com"
```

### Testing Authentication
```python
def test_login():
    response = client.post("/token", data={"username": "testuser", "password": "testpass"})
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_protected_route():
    # Get token
    login_response = client.post("/token", data={"username": "testuser", "password": "testpass"})
    token = login_response.json()["access_token"]
    
    # Access protected route
    response = client.get("/users/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
```

---

## ⚡ Performance & Optimization

### Async Best Practices
```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor

# Don't block the event loop
executor = ThreadPoolExecutor(max_workers=5)

@app.get("/cpu-bound")
async def cpu_bound_task():
    loop = asyncio.get_event_loop()
    # Run CPU-bound task in thread pool
    result = await loop.run_in_executor(executor, cpu_intensive_function)
    return {"result": result}

# Use async libraries
@app.get("/fetch-data")
async def fetch_external_data():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://api.example.com/data') as response:
            return await response.json()

# Concurrent requests
@app.get("/aggregate")
async def aggregate_data():
    tasks = [
        fetch_user_data(1),
        fetch_order_data(1),
        fetch_payment_data(1)
    ]
    results = await asyncio.gather(*tasks)
    return {"user": results[0], "orders": results[1], "payments": results[2]}
```

### Caching
```python
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
from fastapi_cache.backends.redis import RedisBackend

# Initialize cache
@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

# Cache endpoint results
@app.get("/expensive-calculation")
@cache(expire=60)  # Cache for 60 seconds
async def expensive_calculation(param: int):
    # Simulate expensive operation
    await asyncio.sleep(5)
    return {"result": param * 2}

# Manual caching
from functools import lru_cache

@lru_cache(maxsize=128)
def get_settings():
    return Settings()

@app.get("/settings")
async def read_settings(settings: Settings = Depends(get_settings)):
    return settings
```

### Database Connection Pooling
```python
# SQLAlchemy connection pool
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_size=20,           # Number of connections to maintain
    max_overflow=0,         # Maximum overflow connections
    pool_pre_ping=True,     # Test connections before using
    pool_recycle=3600       # Recycle connections after 1 hour
)

# Async connection pool
from databases import Database

database = Database(DATABASE_URL, min_size=10, max_size=20)

@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()
```

### Response Optimization
```python
from fastapi.responses import ORJSONResponse
import orjson

# Use faster JSON serializer
app = FastAPI(default_response_class=ORJSONResponse)

# Streaming responses for large data
from fastapi.responses import StreamingResponse

async def generate_large_csv():
    for i in range(1000000):
        yield f"{i},data_{i}\n"

@app.get("/download-csv")
async def download_csv():
    return StreamingResponse(
        generate_large_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=data.csv"}
    )

# Compression middleware
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

### Pagination
```python
from fastapi import Query

class PaginationParams:
    def __init__(
        self,
        page: int = Query(1, ge=1, description="Page number"),
        size: int = Query(50, ge=1, le=100, description="Page size")
    ):
        self.page = page
        self.size = size
        self.skip = (page - 1) * size

@app.get("/items/")
async def read_items(
    pagination: PaginationParams = Depends(),
    db: Session = Depends(get_db)
):
    items = db.query(Item).offset(pagination.skip).limit(pagination.size).all()
    total = db.query(Item).count()
    
    return {
        "items": items,
        "total": total,
        "page": pagination.page,
        "size": pagination.size,
        "pages": (total + pagination.size - 1) // pagination.size
    }
```

---

## 🚀 Deployment

### Production Server Setup

#### Gunicorn with Uvicorn Workers
```bash
# Install
pip install gunicorn uvicorn[standard]

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# With configuration file (gunicorn.conf.py)
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
loglevel = "info"
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
```

#### Systemd Service
```ini
# /etc/systemd/system/fastapi.service
[Unit]
Description=FastAPI app
After=network.target

[Service]
Type=forking
User=www-data
Group=www-data
WorkingDirectory=/var/www/app
Environment="PATH=/var/www/app/venv/bin"
ExecStart=/var/www/app/venv/bin/gunicorn main:app \
          --workers 4 \
          --worker-class uvicorn.workers.UvicornWorker \
          --bind unix:/var/www/app/fastapi.sock \
          --daemon

[Install]
WantedBy=multi-user.target
```

### Docker Deployment

#### Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Run with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Multi-stage Build
```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Copy dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY . .

# Run with gunicorn
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

#### Docker Compose
```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/dbname
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - ./app:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=dbname
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:alpine

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - web

volumes:
  postgres_data:
```

### Nginx Configuration
```nginx
upstream app {
    server web:8000;
}

server {
    listen 80;
    server_name example.com;

    location / {
        proxy_pass http://app;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /static {
        alias /var/www/static;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fastapi
  template:
    metadata:
      labels:
        app: fastapi
    spec:
      containers:
      - name: fastapi
        image: myregistry/fastapi-app:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: app-secrets
              key: database-url
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: fastapi-service
spec:
  selector:
    app: fastapi
  ports:
    - port: 80
      targetPort: 8000
  type: LoadBalancer
```

---

## 🔧 Advanced Features

### Custom Middleware
```python
import time
from fastapi import Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Class-based middleware
class LoggingMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            print(f"Request to {path}")
        await self.app(scope, receive, send)

app.add_middleware(LoggingMiddleware)
```

### Event Handlers
```python
@app.on_event("startup")
async def startup_event():
    # Initialize database connections
    await database.connect()
    # Load ML models
    app.state.model = load_model()
    # Start background tasks
    asyncio.create_task(periodic_task())

@app.on_event("shutdown")
async def shutdown_event():
    # Close database connections
    await database.disconnect()
    # Clean up resources
    await cleanup()

# Lifespan context manager (newer approach)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await database.connect()
    yield
    # Shutdown
    await database.disconnect()

app = FastAPI(lifespan=lifespan)
```

### Custom Response Classes
```python
from fastapi.responses import Response
import orjson

class ORJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: Any) -> bytes:
        return orjson.dumps(content)

@app.get("/fast-json", response_class=ORJSONResponse)
async def fast_json():
    return {"message": "Fast JSON response"}

# Custom HTML response
from fastapi.responses import HTMLResponse

@app.get("/html", response_class=HTMLResponse)
async def get_html():
    return """
    <html>
        <head><title>FastAPI HTML</title></head>
        <body><h1>Hello from FastAPI!</h1></body>
    </html>
    """
```

### OpenAPI Customization
```python
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="My API",
        version="2.0.0",
        description="This is a custom OpenAPI schema",
        routes=app.routes,
    )
    
    # Add custom x-logo
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"
    }
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

### GraphQL Integration
```python
from strawberry.fastapi import GraphQLRouter
import strawberry

@strawberry.type
class User:
    id: int
    name: str
    email: str

@strawberry.type
class Query:
    @strawberry.field
    async def user(self, id: int) -> User:
        # Fetch user from database
        return User(id=id, name="John Doe", email="john@example.com")

@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_user(self, name: str, email: str) -> User:
        # Create user in database
        return User(id=1, name=name, email=email)

schema = strawberry.Schema(query=Query, mutation=Mutation)
graphql_app = GraphQLRouter(schema)

app.include_router(graphql_app, prefix="/graphql")
```

---

## 💡 Common Interview Questions

### Basic Level

1. **What is FastAPI?**
   - Modern Python web framework
   - Based on Starlette and Pydantic
   - Automatic validation and documentation
   - High performance

2. **FastAPI vs Flask/Django?**
   ```python
   # FastAPI: Type hints, async, auto-docs, fast
   # Flask: Simple, mature, large ecosystem
   # Django: Full-stack, ORM, admin panel
   ```

3. **What are Path Parameters?**
   ```python
   @app.get("/users/{user_id}")
   async def read_user(user_id: int):
       return {"user_id": user_id}
   ```

4. **What are Query Parameters?**
   ```python
   @app.get("/items/")
   async def read_items(skip: int = 0, limit: int = 10):
       return {"skip": skip, "limit": limit}
   ```

5. **How to handle Request Body?**
   ```python
   class Item(BaseModel):
       name: str
       price: float
   
   @app.post("/items/")
   async def create_item(item: Item):
       return item
   ```

### Intermediate Level

1. **Explain Dependency Injection in FastAPI**
   ```python
   # Reusable components
   # Automatic resolution
   # Testing friendly
   # Reduces code duplication
   ```

2. **How does FastAPI validation work?**
   ```python
   # Uses Pydantic models
   # Automatic type conversion
   # Custom validators
   # Error responses
   ```

3. **Async vs Sync endpoints?**
   ```python
   # Async: Non-blocking I/O operations
   @app.get("/async")
   async def async_endpoint():
       await some_async_operation()
       return {"status": "completed"}
   
   # Sync: Blocking operations
   @app.get("/sync")
   def sync_endpoint():
       time.sleep(1)
       return {"status": "completed"}
   ```

4. **How to handle file uploads?**
   ```python
   @app.post("/upload/")
   async def upload_file(file: UploadFile = File(...)):
       contents = await file.read()
       return {"filename": file.filename, "size": len(contents)}
   ```

5. **What is Response Model?**
   ```python
   # Controls response structure
   # Excludes sensitive data
   # Data validation
   # Auto documentation
   ```

### Advanced Level

1. **How does FastAPI handle concurrency?**
   - Uses Starlette's async capabilities
   - Event loop for async operations
   - Thread pool for sync operations
   - Non-blocking I/O

2. **Explain Background Tasks vs Celery**
   ```python
   # Background Tasks: Simple, in-process
   # Celery: Distributed, queue-based, scalable
   ```

3. **How to implement rate limiting?**
   ```python
   from slowapi import Limiter, _rate_limit_exceeded_handler
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
   
   @app.get("/limited")
   @limiter.limit("5/minute")
   async def limited_endpoint(request: Request):
       return {"message": "This is limited"}
   ```

4. **Custom Exception Handling?**
   ```python
   class CustomException(Exception):
       def __init__(self, name: str):
           self.name = name
   
   @app.exception_handler(CustomException)
   async def custom_exception_handler(request: Request, exc: CustomException):
       return JSONResponse(
           status_code=418,
           content={"message": f"Error: {exc.name}"}
       )
   ```

5. **How to optimize FastAPI performance?**
   - Use async operations
   - Connection pooling
   - Caching strategies
   - Response compression
   - Proper indexing
   - Load balancing

### Scenario-Based Questions

1. **Design a REST API for a Blog System**
   ```python
   # Models
   class Post(BaseModel):
       title: str
       content: str
       author_id: int
       tags: List[str] = []
   
   class Comment(BaseModel):
       post_id: int
       content: str
       author_id: int
   
   # Endpoints
   @app.post("/posts/", response_model=Post)
   @app.get("/posts/", response_model=List[Post])
   @app.get("/posts/{post_id}", response_model=Post)
   @app.put("/posts/{post_id}", response_model=Post)
   @app.delete("/posts/{post_id}")
   @app.post("/posts/{post_id}/comments/", response_model=Comment)
   ```

2. **Implement User Authentication System**
   ```python
   # Registration, Login, JWT tokens
   # Password hashing
   # Protected routes
   # Role-based access
   # Token refresh
   ```

3. **Handle High Traffic Load**
   - Use async endpoints
   - Implement caching (Redis)
   - Database connection pooling
   - Horizontal scaling
   - Load balancer (Nginx)
   - CDN for static files

4. **Implement Real-time Chat**
   ```python
   # WebSocket connections
   # Message broadcasting
   # Connection management
   # Message persistence
   # User presence
   ```

5. **Design Microservices Architecture**
   - Service discovery
   - API Gateway
   - Inter-service communication
   - Distributed tracing
   - Circuit breakers

---

## ✅ Best Practices

### Project Structure
```
project/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app
│   ├── config.py         # Configuration
│   ├── models/           # SQLAlchemy models
│   │   ├── __init__.py
│   │   └── user.py
│   ├── schemas/          # Pydantic schemas
│   │   ├── __init__.py
│   │   └── user.py
│   ├── api/              # API routes
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   └── endpoints/
│   │   │       ├── __init__.py
│   │   │       └── users.py
│   ├── core/             # Core functionality
│   │   ├── __init__.py
│   │   ├── security.py
│   │   └── config.py
│   ├── db/               # Database
│   │   ├── __init__.py
│   │   ├── base.py
│   │   └── session.py
│   └── tests/           # Tests
│       ├── __init__.py
│       └── test_users.py
├── alembic/              # Database migrations
├── requirements.txt
├── .env
└── docker-compose.yml
```

### Code Organization
```python
# Separate concerns
# routes/users.py
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

router = APIRouter(prefix="/users", tags=["users"])

@router.get("/")
async def get_users(db: Session = Depends(get_db)):
    return get_all_users(db)

# main.py
from routes import users, posts, auth

app = FastAPI()
app.include_router(users.router)
app.include_router(posts.router)
app.include_router(auth.router)
```

### Configuration Management
```python
# config.py
from pydantic import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "My API"
    database_url: str
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

# Usage
settings = get_settings()
```

### Error Handling
```python
# Custom error responses
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

class DomainException(Exception):
    """Base domain exception"""
    pass

class ItemNotFound(DomainException):
    """Item not found exception"""
    pass

@app.exception_handler(DomainException)
async def domain_exception_handler(request: Request, exc: DomainException):
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.__class__.__name__,
            "message": str(exc)
        }
    )

# Validation error handling
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "details": exc.errors()
        }
    )
```

### Security Best Practices
- 🔒 Always use HTTPS in production
- 🔒 Implement rate limiting
- 🔒 Use secure password hashing (bcrypt)
- 🔒 Validate and sanitize inputs
- 🔒 Use environment variables for secrets
- 🔒 Implement CORS properly
- 🔒 Use JWT with short expiration
- 🔒 Implement request signing for APIs
- 🔒 Log security events
- 🔒 Regular security audits

### Performance Best Practices
- ⚡ Use async for I/O operations
- ⚡ Implement caching strategically
- ⚡ Use connection pooling
- ⚡ Optimize database queries
- ⚡ Use pagination for large datasets
- ⚡ Implement request/response compression
- ⚡ Use CDN for static assets
- ⚡ Monitor and profile regularly
- ⚡ Load test before production
- ⚡ Use appropriate response models

---

## 🔧 Troubleshooting

### Common Issues

#### 1. **Circular Import Errors**
```python
# Problem: Circular dependencies between modules
# Solution: Use TYPE_CHECKING and forward references

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import User

# Or use string annotations
def get_user() -> "User":
    pass
```

#### 2. **Async Context Errors**
```python
# Problem: Using sync code in async function
# Solution: Use async libraries or run_in_executor

# Bad
@app.get("/data")
async def get_data():
    time.sleep(1)  # Blocks event loop
    return {"data": "value"}

# Good
@app.get("/data")
async def get_data():
    await asyncio.sleep(1)
    return {"data": "value"}
```

#### 3. **Database Connection Issues**
```python
# Problem: Connection pool exhaustion
# Solution: Proper session management

# Always close sessions
@app.get("/users/")
async def get_users(db: Session = Depends(get_db)):
    try:
        return db.query(User).all()
    finally:
        db.close()  # Handled by dependency
```

#### 4. **Memory Leaks**
```python
# Problem: Large objects in memory
# Solution: Stream large responses

from fastapi.responses import StreamingResponse

async def generate_large_data():
    for i in range(1000000):
        yield f"{i}\n".encode()

@app.get("/large-data")
async def large_data():
    return StreamingResponse(generate_large_data(), media_type="text/plain")
```

#### 5. **CORS Issues**
```python
# Problem: CORS blocking requests
# Solution: Configure CORS properly

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],  # Don't use ["*"] in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Debugging Tips

#### Enable Debug Mode
```python
# Development only
app = FastAPI(debug=True)

# Better logging
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.get("/debug")
async def debug_endpoint():
    logger.debug("Debug message")
    return {"status": "ok"}
```

#### Request Profiling
```python
import time
from fastapi import Request

@app.middleware("http")
async def profile_request(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} took {process_time:.3f}s")
    return response
```

#### SQL Query Logging
```python
# SQLAlchemy query logging
import logging
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# Or in engine creation
engine = create_engine(
    DATABASE_URL,
    echo=True  # Log all SQL
)
```

---

## 📚 Resources

### Official Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI GitHub](https://github.com/tiangolo/fastapi)
- [Starlette Documentation](https://www.starlette.io/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)

### Learning Resources
- **FastAPI Tutorial** - Official step-by-step guide
- **Full Stack FastAPI PostgreSQL** - Project generator
- **Awesome FastAPI** - Curated list of resources
- **TestDriven.io FastAPI** - TDD with FastAPI

### Books & Courses
- "Building Data Science Applications with FastAPI" - François Voron
- "FastAPI Modern Python Web Development" - Bill Lubanovic
- "Microservice APIs" - José Haro Peralta

### Tools & Extensions
- **FastAPI-Users** - User authentication system
- **FastAPI-SQLAlchemy** - SQLAlchemy integration
- **FastAPI-Cache** - Caching support
- **FastAPI-Limiter** - Rate limiting
- **FastAPI-Mail** - Email support
- **Tortoise-ORM** - Async ORM
- **FastAPI-Admin** - Admin interface
- **FastAPI-MQTT** - MQTT support

### Community
- [FastAPI Gitter](https://gitter.im/tiangolo/fastapi)
- [FastAPI Discord](https://discord.gg/VQjSZaeJmf)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/fastapi)
- [Reddit r/FastAPI](https://www.reddit.com/r/FastAPI/)

---

## 🎯 Quick Reference

### Essential Imports
```python
# Core
from fastapi import FastAPI, Request, Response, status
from fastapi import HTTPException, Depends, Security
from fastapi import Query, Path, Body, Form, File, UploadFile
from fastapi import Header, Cookie

# Responses
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.responses import FileResponse, RedirectResponse

# Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.security import HTTPBearer, HTTPBasic, APIKeyHeader

# Background & WebSocket
from fastapi import BackgroundTasks, WebSocket

# Routing
from fastapi import APIRouter

# Middleware & CORS
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

# Testing
from fastapi.testclient import TestClient

# Pydantic
from pydantic import BaseModel, Field, validator
from pydantic import EmailStr, HttpUrl, SecretStr

# Typing
from typing import Optional, List, Dict, Union, Any
```

### Common Patterns

#### API Versioning
```python
# Version in path
app = FastAPI()

v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

app.include_router(v1_router)
app.include_router(v2_router)

# Version in header
from fastapi import Header

@app.get("/users")
async def get_users(api_version: str = Header(default="v1")):
    if api_version == "v1":
        return {"version": "1.0", "users": [...]}
    elif api_version == "v2":
        return {"version": "2.0", "data": {"users": [...]}}
```

#### Health Checks
```python
@app.get("/health", tags=["health"])
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "service": "my-api",
        "version": "1.0.0"
    }

@app.get("/ready", tags=["health"])
async def readiness_check(db: Session = Depends(get_db)):
    try:
        # Check database connection
        db.execute("SELECT 1")
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Service not ready")
```

#### Request ID Tracking
```python
import uuid
from fastapi import Request

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response

@app.get("/track")
async def track_request(request: Request):
    return {"request_id": request.state.request_id}
```

#### Graceful Shutdown
```python
import signal
import sys

shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    print("Shutting down gracefully...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.on_event("startup")
async def startup():
    asyncio.create_task(monitor_shutdown())

async def monitor_shutdown():
    await shutdown_event.wait()
    # Cleanup tasks
    await cleanup_resources()
    sys.exit(0)
```

---

## 🚦 Performance Monitoring

### APM Integration
```python
# Datadog
from ddtrace import patch_all
patch_all()

# New Relic
import newrelic.agent
newrelic.agent.initialize('newrelic.ini')

# Sentry
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware

sentry_sdk.init(dsn="your-dsn-here")
app.add_middleware(SentryAsgiMiddleware)
```

### Custom Metrics
```python
from prometheus_client import Counter, Histogram, generate_latest
import time

# Metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('app_request_latency_seconds', 'Request latency')

@app.middleware("http")
async def track_metrics(request: Request, call_next):
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path).inc()
    
    start_time = time.time()
    response = await call_next(request)
    REQUEST_LATENCY.observe(time.time() - start_time)
    
    return response

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
```

---

## 🏁 Production Checklist

### Pre-deployment
- [ ] Remove debug mode
- [ ] Set up proper logging
- [ ] Configure CORS properly
- [ ] Use environment variables
- [ ] Set up database migrations
- [ ] Implement health checks
- [ ] Add rate limiting
- [ ] Configure security headers
- [ ] Set up monitoring
- [ ] Load testing completed

### Security
- [ ] HTTPS enabled
- [ ] Authentication implemented
- [ ] Input validation active
- [ ] SQL injection protected
- [ ] XSS prevention
- [ ] CSRF protection (if needed)
- [ ] Secrets encrypted
- [ ] Dependencies updated
- [ ] Security headers configured
- [ ] API versioning

### Performance
- [ ] Database indexed
- [ ] Caching implemented
- [ ] Connection pooling
- [ ] Async operations used
- [ ] Response compression
- [ ] Static files on CDN
- [ ] Query optimization
- [ ] Pagination implemented
- [ ] Rate limiting active
- [ ] Auto-scaling configured

### Monitoring
- [ ] Error tracking (Sentry)
- [ ] APM configured
- [ ] Log aggregation
- [ ] Uptime monitoring
- [ ] Performance metrics
- [ ] Custom dashboards
- [ ] Alerting rules
- [ ] Backup strategy
- [ ] Disaster recovery plan
- [ ] Documentation updated

---

## 📝 Code Snippets

### Custom Validators
```python
from pydantic import validator, root_validator

class User(BaseModel):
    username: str
    email: EmailStr
    age: int
    
    @validator('username')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'must be alphanumeric'
        return v
    
    @validator('age')
    def age_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('must be positive')
        return v
    
    @root_validator
    def check_consistency(cls, values):
        # Cross-field validation
        return values
```

### Dynamic Model Creation
```python
from pydantic import create_model

def create_item_model(fields: dict):
    return create_model('DynamicItem', **fields)

# Usage
ItemModel = create_item_model({
    'name': (str, ...),
    'price': (float, Field(gt=0)),
    'quantity': (int, Field(default=1))
})

@app.post("/dynamic-item/")
async def create_dynamic_item(item: ItemModel):
    return item
```

### Custom Dependency Classes
```python
class PaginationParams:
    def __init__(
        self,
        page: int = Query(1, ge=1),
        size: int = Query(50, ge=1, le=100)
    ):
        self.skip = (page - 1) * size
        self.limit = size

class FilterParams:
    def __init__(
        self,
        q: Optional[str] = None,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None
    ):
        self.q = q
        self.category = category
        self.min_price = min_price
        self.max_price = max_price

@app.get("/products/")
async def get_products(
    pagination: PaginationParams = Depends(),
    filters: FilterParams = Depends()
):
    # Apply pagination and filters
    return {"skip": pagination.skip, "limit": pagination.limit, "filters": filters}
```

---

## 🎉 Conclusion

FastAPI is a powerful, modern web framework that combines the best of Python's type system with high performance and developer-friendly features. This guide covers everything from basic concepts to advanced production deployments.

### Key Takeaways
- 🚀 **Performance** - FastAPI is one of the fastest Python frameworks
- 📝 **Type Safety** - Automatic validation and documentation
- 🔧 **Developer Experience** - Excellent tooling and error messages
- 🏭 **Production Ready** - Used by major companies worldwide
- 🌟 **Active Community** - Regular updates and great support

### Next Steps
1. Build a real project
2. Contribute to open source
3. Share your knowledge
4. Keep learning!

---

<p align="center">
  <b>Happy Coding! 🚀</b><br>
  <i>Remember: The best API is one that's well-documented, fast, and secure!</i>
</p>