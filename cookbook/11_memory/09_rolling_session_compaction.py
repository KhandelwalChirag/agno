from typing import List

from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIResponses
from agno.session.rolling import RollingCompactionManager

# A RollingCompactionManager keeps the recent conversation history verbatim 
# while compressing the older messages into a rolling summary.
rolling_compaction = RollingCompactionManager(
    # After how many uncompacted runs should we trigger a compaction?
    compaction_run_threshold=3,
    # How many of those recent runs should remain uncompacted (verbatim)?
    verbatim_run_budget=1,
)

agent = Agent(
    model=OpenAIResponses(id="gpt-5.5"),
    # Add the rolling_compaction_manager to the agent
    rolling_compaction_manager=rolling_compaction,
    # This must be True to append previous interactions to the model context
    add_history_to_context=True,
    description="You are a helpful memory agent. Keep track of the user's name and details.",
    # We use a memory-based session id for this example
    session_id="rolling_memory_session_1",
)

# Run 1
response: RunResponse = agent.run("Hello! My name is John.")
print(f"Agent: {response.content}")

# Run 2
response = agent.run("I am 30 years old.")
print(f"Agent: {response.content}")

# Run 3 (Threshold reached! Triggers compaction in the background before continuing)
response = agent.run("I work as a software engineer.")
print(f"Agent: {response.content}")

# Check the compacted summary
summary = getattr(agent.session, "rolling_compaction_state", None)
if summary:
    print(f"\nRolling Summary: {summary.summary}")

# Run 4
response = agent.run("What do you know about me?")
print(f"Agent: {response.content}")
