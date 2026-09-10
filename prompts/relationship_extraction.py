def get_relationship_extraction_prompt():
	from langchain_core.prompts import ChatPromptTemplate

	return ChatPromptTemplate.from_template("""Extract meaningful relationships between the supplied entities from the text.
Text:
{text}
Entities:
{entities}
""")
