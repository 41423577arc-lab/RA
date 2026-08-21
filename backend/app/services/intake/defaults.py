DEFAULT_REQUESTER_CONTEXT = {
    "name": "林致远",
    "organization": "澄岳产业发展有限公司",
    "title": "副总经理",
    "role_type": "企业高层领导",
}


def with_default_requester_context(structured_context: dict) -> dict:
    return {
        **structured_context,
        "requester_context": DEFAULT_REQUESTER_CONTEXT,
    }
