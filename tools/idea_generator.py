from smolagents import CodeAgent, Tool, TransformersModel, DuckDuckGoSearchTool, WikipediaSearchTool

# Define your custom tool as a class
class IdeaGenerator(Tool):
    name = "idea_generator"
    description = "Generates creative project ideas based on a given topic and style."
    inputs = {
        "topic": {
            "type": "string",
            "description": "The subject area, e.g., 'AI in cybersecurity'"
        },
        "style": {
            "type": "string",
            "description": "The tone or direction of ideas. Options: 'innovative', 'practical', or 'research'.",
            "nullable": True
        },
    }
    output_type = "string"

    def forward(self, topic: str, style: str = "practical") -> str:
        """
        Core logic of the custom tool.
        """
        return (
            f"Think creatively about '{topic}' and generate three {style} ideas "
            f"that reflect real-world relevance and emerging trends."
        )

# Initialize the model
# Local model path
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

