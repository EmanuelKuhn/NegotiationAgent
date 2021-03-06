# USAGE: python generate_profile.py input_folder output_folder

import argparse
import csv
import json
import os
from dataclasses import dataclass

from generate_domain import *

DOMAIN_NAME="tweedekamer2017kieskompasshortnames"
POSSIBLE_VALUES = ["Helemaal mee eens",
                   "Mee eens",
                   "Neutraal",
                   "Niet mee eens",
                   "Helemaal niet mee eens"]   # maybe scratch "Geen mening" as value?

def generate_for_party(party_name, input_folder, output_folder):
    issues = []

    with open(f'{input_folder}/{party_name}.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        for i, row in enumerate(reader):
            issues.append(Issue(f"stelling{i}", row[1], float(row[2])))

    profile = create_profile(party_name, issues)

    with open(f"{output_folder}/{party_name}.json", mode="w") as json_file:
        json.dump(profile, json_file, indent=True)

    with open(f"{output_folder}/{DOMAIN_NAME}.json", mode="w") as json_file:
        json.dump(generate_domain(DOMAIN_NAME, list(map(lambda i: i.name, issues)), POSSIBLE_VALUES), json_file, indent=True)

@dataclass
class Issue:
    name: str
    value: str # One of {Helemaal mee eens, Mee eens, Neutraal, Niet mee eens, Helemaal niet mee eens, Geen mening}
    weight: float

    def toObj(self):
        obj = {
            self.name: {
                self.name: {
                    "discreteutils": {
                        compute_value_utils(self.value)
                    }
                }
            }
        }

        return obj


def calc_weights(issues):
    weights = {issue.name: round(issue.weight/sum([issue.weight for issue in issues]), 3) for issue in issues}

    sum_of_weights = sum(weights.values())

    for issue in weights.keys():
        weights[issue] += round(1 - sum_of_weights, 3)
        weights[issue] = round(weights[issue], 3)
        break

    sum_of_weights = sum(weights.values())

    assert round(sum_of_weights, 4) == 1.0, sum_of_weights

    return weights


def create_profile(party_name, issues: List[Issue]):
    resultObject = {
        "LinearAdditiveUtilitySpace": {
            "name": party_name,
            "issueUtilities": create_issue_utilities(issues),

            "issueWeights": calc_weights(issues),

            "domain": generate_domain(DOMAIN_NAME, list(map(lambda iss: iss.name, issues)), POSSIBLE_VALUES)
        }
    }

    return resultObject


def create_issue_utilities(issues: List[Issue]):
    issue_utilities = {}

    for issue in issues:
        issue_utilities[issue.name] = compute_value_utils(issue.value)

    return issue_utilities


def create_issue_util(issue: Issue):
    return {
        issue.name: {
            compute_value_utils(issue.value)
        }
    }

    # This should be checked
    # This is somewhat subjective
def compute_value_utils(value: str):
    if value == "Helemaal mee eens":
        return {
            "discreteutils": {
                "valueUtilities": {
                    "Helemaal mee eens": 1.0,
                    "Mee eens": 0.8,
                    "Neutraal": 0.5,
                    "Niet mee eens": 0.3,
                    "Helemaal niet mee eens": 0.1,
                }
            }
        }
    elif value == "Mee eens":
        return {
            "discreteutils": {
                "valueUtilities": {
                    "Helemaal mee eens": 0.8,
                    "Mee eens": 1.0,
                    "Neutraal": 0.7,
                    "Niet mee eens": 0.4,
                    "Helemaal niet mee eens": 0.1,
                }
            }
        }
    elif value == "Neutraal":
        return {
            "discreteutils": {
                "valueUtilities": {
                    "Helemaal mee eens": 0.1,
                    "Mee eens": 0.5,
                    "Neutraal": 1.0,
                    "Niet mee eens": 0.5,
                    "Helemaal niet mee eens": 0.1,
                }
            }
        }
    elif value == "Niet mee eens":
        return {
            "discreteutils": {
                "valueUtilities": {
                    "Helemaal mee eens": 0.1,
                    "Mee eens": 0.4,
                    "Neutraal": 0.7,
                    "Niet mee eens": 1.0,
                    "Helemaal niet mee eens": 0.8,
                }
            }
        }
    elif value == "Helemaal niet mee eens":
        return {
            "discreteutils": {

                "valueUtilities": {
                    "Helemaal mee eens": 0.1,
                    "Mee eens": 0.3,
                    "Neutraal": 0.5,
                    "Niet mee eens": 0.8,
                    "Helemaal niet mee eens": 1.0,
                }
            }
        }
    elif value == "Geen mening":
        return {
            "discreteutils": {

                "valueUtilities": {
                    "Helemaal mee eens": 1.0,
                    "Mee eens": 1.0,
                    "Neutraal": 1.0,
                    "Niet mee eens": 1.0,
                    "Helemaal niet mee eens": 1.0
                }
            }
        }
if __name__ == '__main__':
    party_name = "groenlinks"

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('input_folder',
                        help='Folder of party csv\'s with (stelling, Helemaal mee eens/Mee eens/etc, weight)')
    parser.add_argument('output_folder',
                        help='Folder where the resulting profiles should be stored')

    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    for file in os.listdir(input_folder):
        if not file.endswith(".csv"):
            continue

        party_name = file.split(".csv")[0]

        generate_for_party(party_name, input_folder, output_folder)
