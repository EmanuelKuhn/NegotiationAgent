from typing import List

from generate_profile import Issue


def generate_domain(domain_name: str, issueNames: List[str], possible_values):
    domain = {
        "name": domain_name,
        "issuesValues": dict({(issue, tuple(possible_values)) for issue in issueNames})
    }

    print(domain)

    return domain