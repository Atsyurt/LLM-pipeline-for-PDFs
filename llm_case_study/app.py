from fastapi import FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from src.LanguageModel import LanguageModel


# 14050111012/rag_pipeline_bluecloud
# This class represents a query request with a question attribute of type string.
class QueryRequest(BaseModel):
    question: str


# The code snippet `app = FastAPI()`, `app.mount("/static", StaticFiles(directory="./src/static/"),
# name="static")`, `templates = Jinja2Templates(directory="./src/templates")`, and `lm =
# LanguageModel()` is setting up a FastAPI application with the following components:
app = FastAPI()
app.mount("/static", StaticFiles(directory="./src/static/"), name="static")
templates = Jinja2Templates(directory="./src/templates")

lm = LanguageModel()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    This Python function defines a route for the index page that returns an HTML template with the
    request object passed to it.

    :param request: The `request` parameter in the `index` function represents the incoming HTTP request
    made to the server. It contains information about the request such as headers, cookies, query
    parameters, and more. In this case, it is of type `Request` from the `starlette.requests` module.
    This
    :type request: Request
    :return: The code is returning a response using a template called "index.html" along with the
    request object as context data.
    """
    print("Request for index page received")
    # return templates.TemplateResponse("index.html", {"request": request})
    # return PlainTextResponse("Request for index page received")
    return templates.TemplateResponse("index.html", {"request": request})


# @app.post("/query")
# async def query_llm(request: Request):
#     data = await request.json()
#     name = data.get("name")
#     # show_log_data = data.get("show_log_data", 0)
#     response_text = f"Name: {name}\nNot ready yet."
#     return PlainTextResponse(response_text)


@app.post("/query")
async def query_ai(request: QueryRequest):
    """
    This Python function receives a query request, processes it using a language model, and returns the
    AI response.

    :param request: The `request` parameter in the `query_ai` function is of type `QueryRequest`, which
    is expected to be passed in the request body when making a POST request to the "/query" endpoint.
    This parameter will contain the question that the AI should respond to
    :type request: QueryRequest
    :return: The response from the AI model is being returned as the answer to the question asked in the
    request.
    """
    # ai_response = f"AI Response to: {request.question}"
    ai_response = lm.ask(request.question)

    return {"answer": ai_response}


@app.post("/query_to_HTMLResponse", response_class=HTMLResponse)
async def query_to_HTMLResponse(request: Request, question: str = Form(...)):
    """
    This Python function takes a question as input, generates an AI response using a language model, and
    returns an HTML response with the AI answer displayed on a webpage.

    :param request: The `request` parameter in the function `query_to_HTMLResponse` represents the
    incoming HTTP request made to the server. It contains information about the request such as headers,
    cookies, query parameters, and more. In this case, it is of type `Request` from the
    `starlette.requests`
    :type request: Request
    :param question: The `question` parameter in the `query_to_HTMLResponse` function is a string
    variable that represents the question or query input by the user. This function takes the user's
    question, generates an AI response using a language model ( `lm.ask(question)` does that),
    and then returns an HTML
    :type question: str
    :return: The function `query_to_HTMLResponse` is an endpoint that accepts a POST request with a form
    parameter `question`. It then generates an AI response to the question using the `lm.ask(question)`
    function and returns an HTML response using the `index.html` template. The AI response is passed to
    the template as the `answer` variable.
    """
    # show_log_data = data.get("show_log_data", 0)
    ai_answer = f"AI Response to '{question}'"
    ai_response = lm.ask(question)
    return templates.TemplateResponse(
        "index.html", {"request": request, "answer": ai_response}
    )
