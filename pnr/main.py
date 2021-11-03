from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from cannyeval import  CannyEval  
import uvicorn
app = FastAPI()


@app.get('/')
def index():
    return {"name": "yuvaraj"}


@app.post('/marks')
def pnr(pnr: str):
        evaluatorA = CannyEval()
        report = evaluatorA.report_card(data_json=pnr, max_marks=5, relative_marking=False, integer_marking=True, json_load_version="v2")
       
        return {'valid':report}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)