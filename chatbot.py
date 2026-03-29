import numpy as np
import pickle
import faiss

from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# 1. load data
index = faiss.read_index("data/vecs/vector_index.faiss")

with open("data/vecs/documents.pkl", "rb") as f:
    docs = pickle.load(f)

model = SentenceTransformer("Omartificial-Intelligence-Space/GATE-AraBert-v1")

# 2. load llm
llm = Ollama(model="phi3:mini", temperature=0)

# 3. retrieve context
def retrieve_context(question, k=3):
    query_vector = model.encode([question])
    distances, indices = index.search(np.array(query_vector), k)

    results = []
    for idx in indices[0]:
        results.append(docs[idx])

    return results

# 4. prompt
prompt = ChatPromptTemplate.from_template("""
أنت مساعد ذكي.
أجب اعتمادًا فقط على النص التالي.

قواعد مهمة:
- لا تضف معلومات من عندك
- إذا كانت المعلومات غير كافية، قل: "المعلومات المتاحة غير كافية للإجابة بدقة"
- اجعل الإجابة قصيرة وواضحة وبالعربية

النص:
{context}

السؤال:
{question}

الإجابة:
""")

# 5. main function
def answer_user_question(question):
    context = retrieve_context(question)

    clean_context = [c for c in context if c and len(c.strip()) > 10]
    full_context = "\n".join(clean_context)

    chain = prompt | llm

    response = chain.invoke({
        "context": full_context,
        "question": question
    })

    return response

# 6. run chatbot
if __name__ == "__main__":
    print("Chatbot جاهز!")

    while True:
        question = input("اكتب سؤالك: ")

        answer = answer_user_question(question)

        print("\nالإجابة:")
        print(answer)
        print("=" * 50)