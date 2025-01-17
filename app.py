import os
import gradio as gr
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from duckduckgo_search.exceptions import DuckDuckGoSearchException

# Load environment variables from .env file
load_dotenv()

# Initialize the Groq model with the API key
grok_model = Groq(id="llama-3.1-8b-instant", api_key=os.getenv('groq_api'))

# Initialize the Web Agent with DuckDuckGo as the tool
search_agent = Agent(
    name="Trip Search Agent",
    model=grok_model,
    tools=[DuckDuckGo()],
    instructions=["Provide the best tourist spots, food, and hotels for the given city"],
    show_tool_calls=True,
    markdown=True,
)

def generate_trip_plan(place, days, budget):
    # Construct the prompt to fetch places, food, and hotels
    prompt = f"Create a detailed trip plan for {place} for {days} days with a budget of {budget}. Include:\n"
    prompt += "1. Best places to visit\n"
    prompt += "2. Best food to taste\n"
    prompt += "3. Recommended hotels to stay\n"
    prompt += "4. Ensure all recommendations fit within the budget\n"

    try:
        # Query the agent to get the response
        response = search_agent.print_response(prompt)

        # Process the response to extract relevant information
        trip_plan = {
            'places': "Place 1, Place 2, Place 3",  # Extract from response
            'food': "Food 1, Food 2, Food 3",  # Extract from response
            'hotels': "Hotel 1, Hotel 2, Hotel 3",  # Extract from response
            'budget': budget  # Simply pass the budget as it is
        }

    except DuckDuckGoSearchException as e:
        return f"Search tool rate-limited: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

    # Ensure the full plan is returned
    return f"Trip Plan for {place} ({days} days, budget {budget}):\n" \
           f"Places to visit: {trip_plan['places']}\n" \
           f"Food to taste: {trip_plan['food']}\n" \
           f"Recommended hotels: {trip_plan['hotels']}\n" \
           f"Budget: {trip_plan['budget']}"


# Gradio interface for input and output
def create_trip(place, days, budget):
    return generate_trip_plan(place, days, budget)

# Set up Gradio interface
iface = gr.Interface(
    fn=create_trip,
    inputs=[
        gr.Textbox(label="Place", placeholder="Enter the destination city"),
        gr.Number(label="Days", value=1),  # Set a default value for "Days" if needed
        gr.Number(label="Budget", value=1000),  # Set a default value for "Budget" if needed
    ],
    outputs="text",
    live=True,
    title="Trip Planning Assistant",
    description="Provide a city, number of days, and budget, and get a detailed trip plan with places to visit, food to taste, and hotels to stay within your budget."
)

# Launch Gradio interface
if __name__ == "__main__":
    iface.launch(debug=True)
