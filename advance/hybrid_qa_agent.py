import json
import os
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Dict, List
import csv

from dotenv import load_dotenv
from openai import OpenAI

from ReAct_Agent import create_agent_react_loop, docx_to_markdown
from native_rag import get_query_engine


class HybridQAAgent:
	def __init__(self, max_iterations: int = 100) -> None:
		load_dotenv()

		self._api_key = os.getenv("OPENAI_API_KEY")
		self._api_base = os.getenv("OPENAI_API_BASE")
		self._model_name = os.getenv("OPENAI_API_MODEL")
		self._embed_model_name = os.getenv("OPENAI_API_EMBEDDING_MODEL")

		if not all([self._api_key, self._api_base, self._model_name, self._embed_model_name]):
			raise EnvironmentError("Missing OpenAI environment variables for hybrid agent.")

		self._client = OpenAI(api_key=self._api_key, base_url=self._api_base)
		self._hybrid_iterations = max_iterations

		script_dir = os.path.dirname(os.path.abspath(__file__))
		self._text_docs_dir = os.path.join(script_dir, "datas", "文本数据")
		self._excel_dir = os.path.join(script_dir, "datas", "表数据+表结构说明+表介绍")

		self._text_query_engine = get_query_engine()

		self._context_markdown = self._build_context_markdown()
		self._tools = self._build_tools()

	def _build_context_markdown(self) -> str:
		docx_files = [
			os.path.join(self._excel_dir, "数据库危化数据表结构.docx"),
			os.path.join(self._excel_dir, "数据表介绍.docx"),
		]

		context_parts: List[str] = []
		for path in docx_files:
			if os.path.exists(path):
				markdown = docx_to_markdown(path)
				header = f"## {os.path.basename(path)}"
				context_parts.append(f"{header}\n\n{markdown}")

		return "\n\n".join(context_parts)

	def _build_tools(self) -> List[Dict[str, Any]]:
		return [
			{
				"type": "function",
				"function": {
					"name": "call_text_rag",
					"description": "Use the text RAG pipeline to retrieve unstructured knowledge such as policies or manuals and return an answer.",
					"parameters": {
						"type": "object",
						"properties": {
							"question": {
								"type": "string",
								"description": "The question to answer.",
							}
						},
						"required": ["question"],
					},
				},
			},
			{
				"type": "function",
				"function": {
					"name": "call_tabular_agent",
					"description": "Invoke the structured ReAct agent to analyse Excel tables and return an answer.",
					"parameters": {
						"type": "object",
						"properties": {
							"question": {
								"type": "string",
								"description": "Structured query to resolve with Excel data.",
							}
						},
						"required": ["question"],
					},
				},
			},
			{
				"type": "function",
				"function": {
					"name": "call_dual_mode",
					"description": "Call both text RAG and the structured agent, then return a first-pass fusion of their answers.",
					"parameters": {
						"type": "object",
						"properties": {
							"question": {
								"type": "string",
								"description": "Question that likely needs both modalities.",
							}
						},
						"required": ["question"],
					},
				},
			},
			{
				"type": "function",
				"function": {
					"name": "finish",
					"description": "Call when the final answer is ready.",
					"parameters": {
						"type": "object",
						"properties": {
							"answer": {
								"type": "string",
								"description": "Final answer text.",
							}
						},
						"required": ["answer"],
					},
				},
			},
		]

	def _call_text_rag(self, question: str) -> str:
		return str(self._text_query_engine.query(question)).strip()

	def _call_tabular_agent(self, question: str) -> str:
		buffer = StringIO()
		with redirect_stdout(buffer):
			answer = create_agent_react_loop(
				question=question,
				context_markdown=self._context_markdown,
				excel_dir=self._excel_dir,
				max_iterations=10,
			)
		return (answer or "").strip()

	def _call_dual_mode(self, question: str) -> str:
		text_answer = self._call_text_rag(question)
		table_answer = self._call_tabular_agent(question)
		parts = []
		if text_answer:
			parts.append(f"[Text Retrieval]\n{text_answer}")
		if table_answer:
			parts.append(f"[Tabular Reasoning]\n{table_answer}")
		return "\n\n".join(parts) if parts else "No answer was produced by either path."

	def _execute_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
		if tool_name == "call_text_rag":
			return self._call_text_rag(arguments.get("question", ""))
		if tool_name == "call_tabular_agent":
			return self._call_tabular_agent(arguments.get("question", ""))
		if tool_name == "call_dual_mode":
			return self._call_dual_mode(arguments.get("question", ""))
		if tool_name == "finish":
			return {"finished": True, "answer": arguments.get("answer", "")} 
		return f"未知工具: {tool_name}"

	def answer(self, question: str) -> str:
		system_prompt = (
			"You are a routing and fusion agent."
			" For each question first decide which data modality is required, call the appropriate tools,"
			" and when enough evidence is gathered finish with a concise answer that cites whether it came from text, tables, or both."
		)

		excel_files = [
			f for f in os.listdir(self._excel_dir) if f.lower().endswith(".xlsx")
		]
		excel_paths = [os.path.join(self._excel_dir, f).replace('\\', '/') for f in sorted(excel_files)]
		excel_summary = "\n".join(f"- {path}" for path in excel_paths)

		user_prompt = (
			"Resources:\n"
			"1. Text corpus: datas/文本数据 (vector index ready).\n"
			"2. Tabular data: datas/表数据+表结构说明+表介绍 (Excel).\n"
			"3. Markdown summaries of table schemas are preloaded.\n\n"
			"Excel files:\n"
			f"{excel_summary}\n\n"
			"Question:\n"
			f"{question}\n\n"
			"Follow the ReAct loop: Thought -> Action -> Observation."
		)

		messages: List[Dict[str, Any]] = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		]

		for _ in range(self._hybrid_iterations):
			response = self._client.chat.completions.create(
				model=self._model_name,
				messages=messages,
				tools=self._tools,
				tool_choice="auto",
			)

			assistant_message = response.choices[0].message
			messages.append({
				"role": assistant_message.role,
				"content": assistant_message.content,
				"tool_calls": assistant_message.tool_calls,
			})

			if not assistant_message.tool_calls:
				return assistant_message.content or "Model did not provide an answer."

			for tool_call in assistant_message.tool_calls:
				tool_name = tool_call.function.name
				arguments = json.loads(tool_call.function.arguments)
				observation = self._execute_tool_call(tool_name, arguments)

				if isinstance(observation, dict) and observation.get("finished"):
					return observation.get("answer", "")

				messages.append({
					"tool_call_id": tool_call.id,
					"role": "tool",
					"name": tool_name,
					"content": observation if isinstance(observation, str) else json.dumps(observation),
				})

		return "Maximum iterations reached, no answer determined."

def main() -> None:
    agent = HybridQAAgent()
    
    questions = []
    with open("./question.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                questions.append((row[0], row[1]))

    answers = []
    for qid, question in questions:
        print(f"Processing question {qid}: {question}")
        try:
            answer = agent.answer(question)
            answers.append((qid, answer))
            print(f"Question {qid} completed")
        except Exception as e:
            print(f"Question {qid} failed: {e}")
            answers.append((qid, f"Failed: {str(e)}"))

    with open("answer.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "answer"])
        for qid, answer in answers:
            writer.writerow([qid, answer])

    print(f"Processed {len(answers)} questions, results saved to answer.csv")
if __name__ == "__main__":
	main()
