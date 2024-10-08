# The Individual AI Assistant: (utilities.py)
#
# an LLM chat bot for facilitating solo-person conversation over custom corpus +
# OpenAI corpus (c2021), with local corrective fine-tuning.
#
# The purpose of this bot is to facilitate the universal piece process and make
# the operator's world piece computer less shitty.
#
# License: The Human Imperative...
#
#          ...use this to maintain the universal piece and satisfy The Human Imperative.
#             Failure to do so with mal-intent will result in legal action
#             to protect the service-marked time machine for peace social
#             invention program, under the purview of the federal government
#             for The United States of America. Ask The Individual if you need
#             additional clarification.
#
# This file contains templates and other text content used by The Individual AI assistant
#          
# BEGINS #################################################################################

from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory

import tiktoken
import os.path
import datetime
from random import randint, choice
from rich import print

from server.constants import template, text
from server.constitution import constitution

# logging
DEBUG_LOG_TERSE = "debug_terse.txt"
DEBUG_LOG_VERBOSE = "debug_verbose.txt"

# this is for tuning, determining which similarity search results to filter out
CONTEXT_THRESHOLD = 1
CONTEXT_COUNT = 1
FEEDBACK_THRESHOLD = 1
FEEDBACK_COUNT = 1

class utilities:

    # use this to count tokens for prompt submissions
    def num_tokens(string: str) -> int:

        encoding = tiktoken.encoding_for_model("gpt-4o")
        num_tokens = len(encoding.encode(string))

        return num_tokens

    # this function returns the document id that needs removing when unpoisoning feedback store
    def get_document_id(collection, feedback, feedback_scores):

        # first prepare the feedback
        feedback = feedback.split("====")
        feedback.pop(0)

        # then print feedbacks as numeric list for operator to choose from
        print()
        i = 0
        for entry in feedback:
            print("{}) ".format(i) + "{}".format(feedback_scores[i]))
            print(entry)
            i += 1
        print("\nChoose one of the above feedback chunks to remove:")

        # now get selection, an integer between 0 and len(feedbacks)
        index = int(input())
        if type(index is int) and index < len(feedback) and index >= 0:

            # condition the collection entries so that we can compare docs against selection feedback doc
            i = 0
            for i in range(len(collection["documents"])):
                collection["documents"][i] = collection["documents"][i].replace("\n", "").replace("====", "")

            # search the collection documents for the right feedback doc, and condition feedback text
            global_index = collection["documents"].index(feedback[index].replace("\n", ""))
            # NOTE: removing all line breaks might be completely superfluous

        # if invalid input then break and return to prompt entry
        else:
            print("Your input selection is invalid.")
            return

        # isolate and return embedding id
        return collection["ids"][global_index]

    # this function removes feedback from feedbackdb
    def remove_embedding(vectordb: Chroma, injection: str, scores: list):

        # fetch the embedding collection
        # ...there is no need to include embedding vectors in results
        collection = vectordb.get()

        # we need to find the appropriate document id if we want to replace its contents
        document_id = utilities.get_document_id(collection, injection, scores)

        # we get around the apparent impossibility of a simple embedding extraction
        # by just replacing the document with ""
        # ...because we will not be removing feedback very often, this should not
        #    affect performance
        blank_embedding = vectordb._embedding_function.embed_documents([""])
        vectordb._collection.update(
            ids=[document_id],
            embeddings=blank_embedding,
            documents=[""],
            metadatas=[{"source": ""}]
        )

        return

    # this function is how we add corrective feedback to store
    def add_feedback(feedbackdb: Chroma, original, content, correct):

        # now inject inputs into feedback template
        feedback = template.FEEDBACK.format(original, content, correct)

        # add new feedback text as embedding
        # ...this calls to embedding function to generate new vector
        feedbackdb.add_texts([feedback])

    # this is to combine feedback and context, or whatever other injections we need
    def combined_injections(contextdb: Chroma, feedbackdb: Chroma, prompt):

        context_injection, context_scores = utilities.generate_injection(
            contextdb, prompt, CONTEXT_COUNT, CONTEXT_THRESHOLD)
        feedback_injection, feedback_scores = utilities.generate_injection(
            feedbackdb, prompt, FEEDBACK_COUNT, FEEDBACK_THRESHOLD)
        print("context scores: " + str(context_scores))
        print("feedback scores: " + str(feedback_scores))

        return context_injection, feedback_injection

    # this function returns feedback scores and generates feedback block for injection into template
    def generate_injection(vectordb: Chroma, prompt: str, COUNT, THRESHOLD):

        # get matches for feedback to inject into template
        matches = vectordb.similarity_search_with_score(prompt, distance_metric="cos", k=COUNT)

        # generate feedback text block
        scores = []
        injection = ""
        for vector in matches:

            # isolate only the relevant feedback ... < 0.4 seems to be a sweetspot
            if vector[1] > THRESHOLD:
                continue
            injection = injection + "%s\n" % vector[0].page_content
            scores.append(vector[1])

        return injection, scores

    # we need this helper because we have formatting issues caused by %s vs {}
    # (where the {} approach is used by chain)
    def inject_main(context_injection: str, feedback_injection: str) -> str:

        template_main = template.MAIN % (constitution.CONSTITUTION, context_injection, feedback_injection)

        return template_main

    # we need this helper because we have formatting issues caused by %s vs {}
    # (where the {} approach is used by chain)
    def inject_wild(context_injection: str, topic_1: str, topic_2: str, feedback_injection: str) -> str:

        template_wild = template.WILD % (context_injection, feedback_injection, topic_1, topic_2)

        return template_wild

    # we need this helper because we have formatting issues caused by %s vs {}
    # (where the {} approach is used by chain)
    def inject_child(context_injection: str, latest_response: str, feedback_injection: str) -> str:

        template_child = template.CHILD % (context_injection, feedback_injection, latest_response)

        return template_child

    # we need this helper because we have formatting issues caused by %s vs {}
    # (where the {} approach is used by chain)
    def inject_elder(context_injection: str, latest_response: str, feedback_injection: str) -> str:

        template_elder = template.ELDER % (context_injection, feedback_injection, latest_response)

        return template_elder

    # we need this helper because we have formatting issues caused by %s vs {}
    # (where the {} approach is used by chain)
    def inject_mom(context_injection: str, latest_response: str, feedback_injection: str) -> str:

        template_mom = template.MOM % (context_injection, feedback_injection, latest_response)

        return template_mom

    # we need this helper because we have formatting issues caused by %s vs {}
    # (where the {} approach is used by chain)
    def inject_teen(context_injection: str, latest_response: str, feedback_injection: str) -> str:

        template_teen = template.TEEN % (context_injection, feedback_injection, latest_response)

        return template_teen

    # we need this helper because we have formatting issues caused by %s vs {}
    # (where the {} approach is used by chain)
    def inject_simple(context_injection: str, latest_response: str, feedback_injection: str) -> str:

        template_simple = template.SIMPLE % (context_injection, feedback_injection, latest_response)

        return template_simple

    # we need this helper because we have formatting issues caused by %s vs {}
    # (where the {} approach is used by chain)
    def inject_friend(context_injection: str, latest_response: str, feedback_injection: str) -> str:

        template_friend = template.FRIEND % (context_injection, feedback_injection, latest_response)

        return template_friend

    # this function gets two random topics from menu
    def choose_topics():
        topic_1 = choice(text.TOPICS)
        topic_2 = choice(text.TOPICS)

        return topic_1, topic_2

    # function to log each exchange for debuggin purposes
    # ...we will need different logs for collecting training data
    def log_exchange(
        prompt: str,
        response,
        context_scores,
        context,
        feedback_scores,
        feedback,
        tokens
    ):

        # generate ID number for log entry so we can refer easily to verbose log entries by terse index
        entry_id = randint(1000, 9999)

        original_text = []
        # let's stack this exchange onto the log
        for log in [DEBUG_LOG_TERSE, DEBUG_LOG_VERBOSE]:
            if not os.path.exists(log):
                append_copy = open(log, "x")
                original_text.append("")
            else:
                append_copy = open(log, "r")
                original_text.append(append_copy.read())
            append_copy.close()

        # now clobber the shit out of these puny litte original logs     !
        # TODO (this is, er, a little cumbersome and inefficient...could use some work on logging)

        # terse log
        append_copy = open(DEBUG_LOG_TERSE, "w")
        append_copy.write("BEGIN LOG ENTRY, ID={} ============= ".format(entry_id) +
                          datetime.datetime.now().strftime("%a %d %B %Y ~ %H:%M") + "\n")
        append_copy.write("Human: \n")
        append_copy.write(prompt + "\n")
        append_copy.write("The Individual: \n")
        append_copy.write(response["response"] + "\n")
        append_copy.write("END LOG ENTRY,   ID={} ============= ".format(entry_id) + "\n\n")
        append_copy.write(original_text[0])
        append_copy.close()

        # verbose log
        append_copy = open(DEBUG_LOG_VERBOSE, "w")
        append_copy.write("BEGIN LOG ENTRY, ID={} ============= ".format(entry_id) +
                          datetime.datetime.now().strftime("%a %d %B %Y ~ %H:%M") + "\n")
        append_copy.write("Human: \n")
        append_copy.write(prompt + "\n")
        append_copy.write("The Individual: \n")
        append_copy.write(response["response"] + "\n")
        append_copy.write("Context scores: \n")
        append_copy.write(str(context_scores) + "\n")
        append_copy.write("Context: \n")
        append_copy.write(context + "\n")
        append_copy.write("Feedback scores: \n")
        append_copy.write(str(feedback_scores) + "\n")
        append_copy.write("Feedback: \n")
        append_copy.write(context + "\n")
        append_copy.write("Total tokens: \n")
        append_copy.write(str(tokens) + "\n")
        append_copy.write("END LOG ENTRY,   ID={} ============= ".format(entry_id) + "\n\n")
        append_copy.write(original_text[1])
        append_copy.close()

# ENDS ###################################################################################
# .
# .
# .
# _we need to erect a global peace system_ - tW
