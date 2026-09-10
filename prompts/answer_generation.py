def get_answer_prompt():
	from langchain_core.prompts import ChatPromptTemplate

	return ChatPromptTemplate.from_template("""Answer the user question using only the retrieved evidence below.
Do not invent facts or assume unsupported information. If evidence is insufficient, say so.
Be concise and accurate. Do not claim to have searched a source that was not retrieved.
Question: {question}
Evidence:
{context}
""")
