from fastapi import FastAPI, Form, Request, status
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from src.LanguageModel import LanguageModel


class QueryRequest(BaseModel):
    question: str


app = FastAPI()
app.mount("/static", StaticFiles(directory="./src/static/"), name="static")
templates = Jinja2Templates(directory="./src/templates")

lm = LanguageModel("./required_Sources/gemma-3-4b-it-q4_0.gguf")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
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
    # ai_response = f"AI Response to: {request.question}"
    ai_response = lm.ask(request.question)

    return {"answer": ai_response}


@app.post("/query_to_HTMLResponse", response_class=HTMLResponse)
async def query_to_HTMLResponse(request: Request, question: str = Form(...)):
    # show_log_data = data.get("show_log_data", 0)
    ai_answer = f"AI Response to '{question}'"
    ai_response = lm.ask(question)
    return templates.TemplateResponse(
        "index.html", {"request": request, "answer": ai_response}
    )


# if name:
#     print("Request for hello page received with name=%s" % name)
#     name, log = run_ai_stream(name, 1, True)

#     if show_log_data == 1:
#         response_text = f"Name: {name}\nReasoning Steps:\n{log}"
#     else:
#         response_text = f"Name: {name}\nReasoning Steps: Not included"

#     return PlainTextResponse(response_text)
# else:
#     print(
#         "Request for hello page received with no name or blank name -- redirecting"
#     )
#     return PlainTextResponse("Redirecting to index page.")


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=80)
