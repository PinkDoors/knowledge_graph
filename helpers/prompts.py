import sys
from yachalk import chalk
sys.path.append("..")

import json
from ollama import client

def extractConcepts(prompt: str, metadata={}, model="mistral-openorca:latest"):
    SYS_PROMPT = (
        "Your task is extract the key concepts (and non personal entities) mentioned in the given context. "
        "Extract only the most important and atomistic concepts, if  needed break the concepts down to the simpler concepts."
        "Categorize the concepts in one of the following categories: "
        "[event, concept, place, object, document, organisation, condition, misc]\n"
        "Format your output as a list of json with the following format:\n"
        "[\n"
        "   {\n"
        '       "entity": The Concept,\n'
        '       "importance": The concontextual importance of the concept on a scale of 1 to 5 (5 being the highest),\n'
        '       "category": The Type of Concept,\n'
        "   }, \n"
        "{ }, \n"
        "]\n"
    )
    # client.BASE_URL = "http://host.docker.internal:11434"
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=prompt)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result


def graphPrompt(input: str, metadata={}, model="mistral-openorca:latest"):
    if model == None:
        model = "mistral-openorca:latest"

    # model_info = client.show(model_name=model)
    # print( chalk.blue(model_info))

    SYS_PROMPT = (
        "You are a knowledge graph generator who extracts terms and their relations from a given context.You are provided with a context chunk (delimited by ```). "
        "Your task is to extract a fully connected ontology of terms mentioned in the given context. "
        "These terms should represent the key concepts as per the context.\n"
        "GLOBAL CONSTRAINTS (MUST BE STRICTLY FOLLOWED):\n "
        "- The knowledge graph MUST be fully connected.\n"
        "- There MUST be exactly ONE root node representing the main concept of the context.\n"
        "- Every other node MUST be directly or indirectly connected to the root node.\n"
        "- Each node MUST have at least ONE relation to an already existing node.\n"
        "- NO isolated nodes and NO isolated subgraphs are allowed.\n"
        "- Do NOT generate standalone terms without relations.\n"
        "- Every newly introduced node MUST be linked to at least one previously introduced node.\n"
        "Thought 1: While traversing through each sentence, Think about the key terms mentioned in it.\n"
            "\tTerms may include object, entity, location, organization, person, \n"
            "\tcondition, acronym, documents, service, concept, etc.\n"
            "\tTerms should be as atomistic as possible\n\n"
        "Thought 2: Do NOT generate a separate list of terms. "
        "Instead, think ONLY in terms of RELATIONS (triples). "
        "Every concept MUST appear for the first time only as part of a relation.\n\n"
        "Thought 3: For every new term, explicitly decide:\n "
        "- Which existing term it must be connected to\n"
        "- Why this connection is logically necessary\n"
        "New terms that cannot be connected MUST NOT be generated.\n\n"
        "Thought 4: Generate the knowledge graph so that a student can learn each concept sequentially. "
        "Ensure that no concept refers to or depends on a concept that has NOT been introduced earlier. "
        "Each next relation must extend the already connected graph.\n\n"
        "Thought 5: Enforce global reachability: "
        "From ANY node in the graph, it must be possible to reach ANY other node by following relations. "
        "If at any moment a disconnected component is about to appear, you MUST instead connect it to the closest existing concept in the main graph.\n\n"
        "OUTPUT FORMAT:\n"
        "Format your output strictly as a JSON list of relations.Each element of the list contains exactly one pair of connected terms and their relation:\n"
        "Example output (Don't include any other annotations, start with [ symbol immediately):\n"
        "[\n"
        "   {\n"
        '       "node_1": "A concept from extracted ontology",\n'
        '       "node_2": "A related concept from extracted ontology",\n'
        '       "edge": "relationship between the two concepts, node_1 and node_2 in one or two sentences"\n'
        "   }, \n"
        "{ }, \n"
        "]"
    )

    # client.BASE_URL = "http://host.docker.internal:11434"
    USER_PROMPT = f"context: ```{input}``` \n\n output: "
    response, _ = client.generate(model_name=model, system=SYS_PROMPT, prompt=USER_PROMPT)
    try:
        result = json.loads(response)
        result = [dict(item, **metadata) for item in result]
    except:
        print("\n\nERROR ### Here is the buggy response: ", response, "\n\n")
        result = None
    return result
