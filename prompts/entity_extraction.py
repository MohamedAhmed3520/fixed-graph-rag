def get_entity_extraction_prompt():
	from langchain_core.prompts import ChatPromptTemplate

	return ChatPromptTemplate.from_template("""Extract meaningful entities from the text. Return only entities supported by the text.
Text:
{text}
""")
