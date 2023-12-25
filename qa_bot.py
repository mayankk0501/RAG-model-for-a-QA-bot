import openai
import pinecone
from sklearn.feature_extraction.text import TfidfVectorizer
import json

openai.api_key = ""
pinecone_api_key = ""
pinecone.init(api_key=pinecone_api_key, environment="gcp-starter")


faq_file_path = "Ecommerce_FAQ_Chatbot_dataset.json"
with open(faq_file_path, 'r') as file:
    faq_data = json.load(file)

questions = [entry["question"] for entry in faq_data["questions"]]
answers = [entry["answer"] for entry in faq_data["questions"]]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions).toarray()

index_name = "qa-bot"
index = pinecone.Index(index_name)
pinecone.create_index(name = index_name, dimension = len(question_vectors[0]))

vectors = [
    (
        str(i),
        vector.tolist(),
        {"question": question, "answer": answer}
    ) for i, (vector, question, answer) in enumerate(zip(question_vectors, questions, answers))
]

upsert_response = index.upsert(vectors=vectors)

def rag_model_with_openai(user_question):
    user_question_vector = vectorizer.transform([user_question]).toarray()[0]
    response = index.query(user_question_vector.tolist(), top_k=3, include_metadata=True)
    metadata = response['matches'][0]['metadata']

    if response['matches'][0]['score'] > 0.7:
        original_answer = metadata['answer']
    else:
        original_answer = 'The answer is not available in FAQs.'

    qa = ''
    ques = []
    for i in response['matches']:
        qa += f"Q: {i['metadata']['question']} A: {i['metadata']['answer']}     "
        ques.append(i['metadata']['question'])

    openai_prompt = f"Given the question: '{user_question}' and most precise answer: {original_answer} and relevant Q&A pairs: '{qa}', provide a concise and accurate answer based on the information given."

    refined_answer = openai.Completion.create(
        engine="text-davinci-003",
        prompt=openai_prompt,
        temperature=0.5,
        max_tokens=150
    )

    refined_answer = refined_answer['choices'][0]['text'].strip()
    return refined_answer, ques

user_question = input('How can I help you?\nQuestion: ')
while True:
    refined_answer, ques = rag_model_with_openai(user_question)
    print(f"Answer: {refined_answer}")
    if refined_answer == 'The answer is not available in FAQs.':
        print('Try asking: ')
        for i in ques:
            print(i)
    print('Any Other Question? Y/N')
    x = input()
    if x == 'n' or x == 'N':
        break
    else:
        user_question = input('Question: ')

