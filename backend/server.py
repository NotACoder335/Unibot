from fastapi import FastAPI  
from pydantic import BaseModel
try:
    from .rag import ask
except ImportError:
    from rag import ask
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class Question(BaseModel):
    question: str


@app.post("/ask")
def askp(question: Question):
    answer = ask(question.question)

    return {
        "answer": answer
    }

def test_q():

    '''
    run to check if model is working. comment everything else while running this fn..
    '''
    user_question = "Who is the prme miniter of India?"

    print("\n" + "=" * 70)
    print("🎓 ADMISSION ENQUIRY CHATBOT - USER QUERY")
    print("=" * 70)
    print(f"\nStudent Question: {user_question}")
    print("\n" + "-" * 70)
    print("Response:")
    print("-" * 70)
    result = ask(user_question)
    print(result)
    print("\n" + "=" * 70)
