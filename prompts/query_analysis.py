from pydantic import BaseModel, Field


class QueryAnalysis(BaseModel):
    rewritten_question: str
    entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


def get_query_analysis_prompt():
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_template("""Analyze this question for retrieval. Keep the original meaning. Identify named entities and useful keywords.
Question: {question}
""")
