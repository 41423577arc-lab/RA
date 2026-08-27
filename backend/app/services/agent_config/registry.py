from dataclasses import dataclass


@dataclass(frozen=True)
class NodeSpec:
    node_key: str
    output_schema: str
    conditional: bool = False
    allows_tools: bool = False


NODE_REGISTRY = {
    item.node_key: item
    for item in (
        NodeSpec("intake_chat", "IntakeChatResult"),
        NodeSpec("intake_agent", "AgentTurn", allows_tools=True),
        NodeSpec("intake_identity_initialize", "IntakeStructuredContext"),
        NodeSpec("intake_identity_update", "IntakeStructuredContext", allows_tools=True),
        NodeSpec("intake_followup", "IntakeFollowupResult", conditional=True),
        NodeSpec(
            "intake_identity_normalize",
            "ExternalIdentityNormalizationResult",
            conditional=True,
        ),
        NodeSpec("intake_readiness", "IntakeReadinessResult", conditional=True),
        NodeSpec("intake_final_confirmation", "IntakeFinalConfirmationResult"),
        NodeSpec("evidence_verify", "WebEvidenceDecision", conditional=True),
        NodeSpec("final_synthesis", "GeneratedReportContent"),
        NodeSpec("analysis_chat", "TaskChatResult", conditional=True),
    )
}
