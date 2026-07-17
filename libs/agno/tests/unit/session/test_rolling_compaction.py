from typing import List
from agno.agent import Agent
from agno.session.rolling import RollingCompactionManager, RollingCompactionState
from agno.session.agent import AgentSession
from agno.run.agent import RunOutput
from agno.models.message import Message
from agno.session.summary import SessionSummaryManager

def test_rolling_compaction_manager_initialization():
    manager = RollingCompactionManager(compaction_run_threshold=5)
    assert manager.compaction_run_threshold == 5
    assert manager.model is None

def test_rolling_compaction_state():
    state = RollingCompactionState(
        summary="Old summary",
        compacted_through_run_id="run_123",
        version=1
    )
    assert state.summary == "Old summary"
    assert state.compacted_through_run_id == "run_123"
    assert state.version == 1

def test_agent_initialization_with_rolling_compaction():
    agent = Agent(
        rolling_compaction_manager=RollingCompactionManager(compaction_run_threshold=3, verbatim_run_budget=1)
    )
    assert agent.rolling_compaction_manager is not None
    assert agent.rolling_compaction_manager.compaction_run_threshold == 3

def test_agent_mutual_exclusion():
    from agno.agent._init import set_rolling_compaction_manager
    # If both are provided, session_summary_manager should be set to None during init
    agent = Agent(
        rolling_compaction_manager=RollingCompactionManager(compaction_run_threshold=3, verbatim_run_budget=1),
        session_summary_manager=SessionSummaryManager()
    )
    set_rolling_compaction_manager(agent)
    assert agent.rolling_compaction_manager is not None
    assert agent.session_summary_manager is None

def test_should_compact():
    from agno.run.agent import RunOutput
    
    manager = RollingCompactionManager(compaction_run_threshold=3, verbatim_run_budget=1)
    class DummySession:
        runs = []
    
    session = DummySession()
    # No runs, shouldn't compact
    assert not manager.should_compact(session)
    
    # Add runs up to threshold
    session.runs = [RunOutput(run_id=f"run_{i}") for i in range(3)]
    assert manager.should_compact(session)
    
    # Set compaction state
    session.rolling_compaction_state = RollingCompactionState(
        summary="summary",
        compacted_through_run_id="run_1",
        version=1
    )
    # The start_idx should be 2 (runs[2] and onwards). Only 1 uncompacted run left.
    assert not manager.should_compact(session)
