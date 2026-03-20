from smolagents import CodeAgent, TransformersModel, DuckDuckGoSearchTool, WikipediaSearchTool
from ..tools.IdeaGenerator import IdeaGenerator


local_model_path = "/home/cs/models/qwen2.5-coder-7b-model"  
model = TransformersModel(model_id=local_model_path)

agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        WikipediaSearchTool(),
        IdeaGenerator(),
    ],
    model=model
)

# Ask a question that requires reasoning across multiple tools
query = "Find the latest AI breakthroughs in cybersecurity research and generate three innovative project ideas."


result = agent.run(query, max_steps=10)

print(result)
