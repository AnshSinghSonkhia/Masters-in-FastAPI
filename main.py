# fastapi is a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints. It is built on top of Starlette for the web parts and Pydantic for the data parts.
from fastapi import FastAPI

# pydantic is a data validation and settings management library for Python, used with FastAPI for request and response models.
from pydantic import BaseModel

# typing is a standard library module that provides support for type hints.
from typing import List

app = FastAPI()     # Create an instance of the FastAPI class

# Define a Pydantic model for the tea item
class Tea(BaseModel):
    id: int
    name: str
    origin: str

teas: List[Tea] = []  # Initialize an empty list to store tea items

# Decorators are used to define the route and the HTTP method

# Home route
@app.get("/")
def read_root():
    return {"message": "Welcome to the Tea API!"}

# Route to add a new tea item
@app.get("/teas")
def get_teas():
    """Get all teas."""
    return teas

# Route to add a new tea item
@app.post("/teas")
def add_tea(tea: Tea):
    """Add a new tea."""
    teas.append(tea)
    return {"message": "Tea added successfully!", "tea": tea}

# Route to get a specific tea by ID
@app.put("/teas/{tea_id}")
def update_tea(tea_id: int, updated_tea: Tea):
    """Update an existing tea."""
    for index, tea in enumerate(teas):
        if tea.id == tea_id:
            teas[index] = updated_tea
            return {"message": "Tea updated successfully!", "tea": updated_tea}
    return {"error": "Tea not found."}

# Route to delete a tea by ID
@app.delete("/teas/{tea_id}")
def delete_tea(tea_id: int):
    """Delete a tea by ID."""
    for index, tea in enumerate(teas):
        if tea.id == tea_id:
            deleted = teas.pop(index)
            return {"message": "Tea deleted successfully!", "tea": deleted}
    return {"error": "Tea not found."}