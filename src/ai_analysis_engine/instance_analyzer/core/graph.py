"""
Main LangGraph workflow for AI Analysis Engine
"""

from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..models.state import AnalysisState
from ..config import config
from ..utils.logger import get_logger
from .nodes import (
    InitializationNode,
    SupervisorNode,
    DataCheckerNode,
    ConsistencyCheckerNode,
    HypothesisGeneratorNode,
    VerifierNode,
    ReporterNode
)

logger = get_logger(__name__)


class AnalysisGraph:
    """
    Main workflow graph for the AI Analysis Engine
    """

    def __init__(self):
        self.graph = None
        self.nodes = {}
        self._initialize_nodes()

    def _initialize_nodes(self):
        """Initialize all workflow nodes"""
        self.nodes = {
            "initialization": InitializationNode(),
            "supervisor": SupervisorNode(),
            "data_checker": DataCheckerNode(),
            "consistency_checker": ConsistencyCheckerNode(),
            "hypothesis_generator": HypothesisGeneratorNode(),
            "verifier": VerifierNode(),
            "reporter": ReporterNode()
        }

        logger.info("Initialized all workflow nodes")

    def build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow

        Returns:
            Compiled StateGraph
        """
        # Create the graph with our state model
        workflow = StateGraph(AnalysisState)

        # Add nodes
        for node_name, node_instance in self.nodes.items():
            workflow.add_node(node_name, node_instance.process)

        # Set entry point
        workflow.set_entry_point("initialization")

        # Add edges
        workflow.add_edge("initialization", "supervisor")

        # Supervisor routes to different nodes based on state
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "data_checker": "data_checker",
                "consistency_checker": "consistency_checker",
                "hypothesis_generator": "hypothesis_generator",
                "verifier": "verifier",
                "reporter": "reporter",
                END: END
            }
        )

        # Data checker can go back to supervisor or to consistency checker
        workflow.add_edge("data_checker", "supervisor")

        # Consistency checker goes back to supervisor
        workflow.add_edge("consistency_checker", "supervisor")

        # Hypothesis generator goes back to supervisor
        workflow.add_edge("hypothesis_generator", "supervisor")

        # Verifier can loop back to hypothesis generator or go to reporter
        workflow.add_conditional_edges(
            "verifier",
            self._route_from_verifier,
            {
                "hypothesis_generator": "hypothesis_generator",
                "reporter": "reporter"
            }
        )

        # Reporter is the final node
        workflow.add_edge("reporter", END)

        # Compile the graph without checkpointer to avoid serialization issues
        self.graph = workflow.compile()
        logger.info("Built and compiled LangGraph workflow")

        return self.graph

    def _route_from_supervisor(self, state: AnalysisState) -> str:
        """
        Route from supervisor based on current state

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        current_dataset = state.get_current_dataset()

        if current_dataset is None:
            # No more datasets to process
            return END

        if current_dataset.status == "pending":
            return "data_checker"
        elif current_dataset.status == "data_checked":
            return "consistency_checker"
        elif current_dataset.status == "consistency_checked":
            return "hypothesis_generator"
        elif current_dataset.status == "hypothesis_generated":
            return "verifier"
        elif current_dataset.status == "verified":
            return "reporter"
        else:
            # Default to data checker for unknown states
            return "data_checker"

    def _route_from_verifier(self, state: AnalysisState) -> str:
        """
        Route from verifier based on verification results

        Args:
            state: Current workflow state

        Returns:
            Next node name ("hypothesis_generator" or "reporter")
        """
        current_dataset = state.get_current_dataset()

        # Report regardless of success to avoid infinite loops; the report will reflect outcome
        return "reporter"

    def run_analysis(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete analysis workflow

        Args:
            initial_state: Initial state dictionary

        Returns:
            Final state after workflow completion
        """
        if self.graph is None:
            self.build_graph()

        try:
            # Convert dict to AnalysisState if needed
            if isinstance(initial_state, dict):
                state = AnalysisState(**initial_state)
            else:
                state = initial_state

            logger.info("Starting analysis workflow")

            # Convert to dict for LangGraph (avoids serialization issues)
            state_dict = state.model_dump()

            # Run the workflow
            result = self.graph.invoke(state_dict, config={"configurable": {"thread_id": "analysis_workflow"}})

            logger.info("Analysis workflow completed")

            # Convert result back to AnalysisState if needed
            if isinstance(result, dict):
                # Convert datasets back to DatasetInfo objects
                if "datasets" in result:
                    from ..models.state import DatasetInfo
                    result["datasets"] = [
                        DatasetInfo(**ds) if isinstance(ds, dict) else ds
                        for ds in result["datasets"]
                    ]

            return result

        except Exception as e:
            logger.error(f"Analysis workflow failed: {e}")
            raise

    def get_graph_visualization(self) -> str:
        """
        Get mermaid visualization of the graph

        Returns:
            Mermaid graph string
        """
        if self.graph is None:
            self.build_graph()

        # This is a simplified visualization
        # In a real implementation, you might use graph.get_graph() or similar
        mermaid_graph = """
        graph TD
            START[Start] --> Init[Initialization]
            Init --> Supervisor[Supervisor]
            Supervisor --> DataChecker[Data Checker]
            Supervisor --> ConsistencyChecker[Consistency Checker]
            Supervisor --> HypothesisGenerator[Hypothesis Generator]
            Supervisor --> Verifier[Verifier]
            Supervisor --> Reporter[Reporter]
            DataChecker --> Supervisor
            ConsistencyChecker --> Supervisor
            HypothesisGenerator --> Supervisor
            Verifier --> HypothesisGenerator
            Verifier --> Reporter
            Reporter --> END[End]
        """

        return mermaid_graph.strip()
