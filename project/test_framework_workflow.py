import unittest

from smolagents import ToolCallingAgent

from project_starter import ProjectOrchestrator


class FrameworkWorkflowTests(unittest.TestCase):
    def test_manager_delegates_quote_to_registered_framework_specialist(self):
        orchestrator = ProjectOrchestrator()

        self.assertIsInstance(orchestrator.agent, ToolCallingAgent)
        self.assertEqual(
            set(orchestrator.agent.managed_agents),
            {
                "inventory_specialist",
                "quote_specialist",
                "fulfillment_specialist",
                "business_advisor",
            },
        )
        self.assertLessEqual(len(orchestrator.agent.managed_agents) + 1, 5)
        self.assertTrue(
            all(
                isinstance(agent, ToolCallingAgent)
                for agent in orchestrator.agent.managed_agents.values()
            )
        )

        quote = orchestrator._delegate(
            "quote_specialist",
            "create_quote",
            {
                "lines": [
                    {
                        "item_name": "A4 paper",
                        "quantity": 500,
                        "unit_price": 0.05,
                    }
                ],
                "customer_brief": {},
            },
        )

        self.assertEqual(quote["subtotal"], 25.0)
        self.assertEqual(quote["discount"], 1.25)
        self.assertEqual(quote["total"], 23.75)
        self.assertTrue(
            any(
                call.name == "quote_specialist"
                for step in orchestrator.agent.memory.steps
                for call in getattr(step, "tool_calls", [])
            )
        )
        self.assertEqual(orchestrator.workflow_model.call_number, 4)


if __name__ == "__main__":
    unittest.main()
