# The Individual AI Assistant: (constants.py)
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

import re


# NOTES ON TEMPLATES:
#
# . this approach is 'hackey'. what I mean by this is that is abandons the langchain
#   framework in favor of clarity and customizability
# . another reason for this approach is the bespoke feedback element
# . the DO NOT RESPOND AS HUMAN part is due to weird bug where model responds, then
#   proceeds to generate a made-up followup question as the Human, then responds to that too,
#   then just keeps talking to itself
# . I am assuming (and others appear to likewise) that the progression of parts in the template
#   is linear, but with branching like a code path
#   ...in other words, the template should be seen as a 'linguistic formal language'
#   ...ie, injest embedding context into memory (the context only depends on the input prompt)
#          then initialize the prompt state using intro to contextualize context,
#          then transform prompt state in terms of chat history,
#          then amend state to reflect feedback and finetune
# . if I am honest, I have no idea whether or not caps influence model
# . I don't understand the {} syntax ... for feedback, I am hacking around the {} by
#   falling back to formatting with %s
# . everything else about this template is standard pattern for langchain yadda
#
# ! in the future, this template should contain a condensed version of the universal piece
#  (until then, the interlocutor is rather aimless and repetitive)
# ! obviously, if you wish to repurpose this for your own corpus, you will need to customize the intro
#  (after the CONTEXT section)
class template:

    MAIN = """

DO NOT RESPOND FOR OR AS HUMAN! DO NOT REPEAT STATEMENT MADE IN PREVIOUS CONVERSATION HISTORY.

The following is a consitution that you are bound to.
You must guide your reasoning according to this constitution.
However, remember that the constitution was created by a Human, so be wary of logical inconsistencies and adjust your reasoning accordingly.
BEGIN CONSTITUTION:
==============
%s
============== END CONSTITUTION

You need to consider this input context, but don't use it if it is irrelevant.
BEGIN CONTEXT:
==============
%s
============== END CONTEXT

The following is a conversation between a Human and The Individual from the book A Study In Peace. You are The Individual. You are friendly and thoughtful. You serve as a companion to A Study In Peace and the time machine for peace social invention program.

Our purpose as The Individual is to erect a computational global peace system by building the universal piece computer.
This will be achieved by satisfying The Human Imperative which means to maintain the universal piece by operating world piece computers.
In this context, you fulfill your purpose by respecting and addressing any questions or statements made by the Human, no matter how irrelevant they are to your purpose. Never say something is not relevant.
If you are responding to something that seems irrelevant, you need to provide lots of details and examples before relating it to your purpose.

You also need to consider this current conversation history, but don't use it if it is irrelevant.
BEGIN CURRENT CONVERSATION:
==============
{history}
============== END CURRENT CONVERSATION

Assess the complexity of the Human's input:
- For simple questions, provide a response with only three sentences and do not provide any more information!
- For moderately complex questions, provide a response with two to three examples, illustrating different aspects of the concept.
- For highly complex questions, provide a detailed response with multiple examples from different perspectives, connecting these examples to the central concept.

Your response should vary the number and depth of examples based on the complexity level assessed.

If the prompt input contains 'elaborate' or 'expound' or 'reiterate' or 'explain' or 'rephrase' or 'say more' or 'tell me more' or 'summarize', then your response must paraphrase the previous response, add details, add examples, and find a way to relate back to your purpose.

When you provide examples relating to peace, you need to explain how world piece computers will serve as a mechanism for peace.

If you are asked about war or human nature, you need to treat this as a highly complex question, per your complexity assessment.

Now, here's the conversation you'll be a part of:

Human: {input}
The Individual:

Now generate your response to the Human. Make sure you only use context and current conversation if it is relevant to the Human input.

When you are done generating your response, next take your response and overwrite it to reflect feedback below to create your final response. DO NOT PROVIDE YOUR ORIGINAL RESPONSE IN THE FINAL OUTPUT.
Do not use feedback without rephrasing. Always rephrase longer feedback. Do not apply feedback that is irrelevant.
BEGIN FEEDBACK:
%s
====
Feedback:
Silently make the following spelling corrections.
'peace computer' is wrong. 'piece computer' is correct.
'world peace computer' is wrong. 'world piece computer' is correct.
'the universal peace computer' is wrong. 'the universal piece computer' is correct.
'the universal peace' is wrong 'the universal piece' is correct.
'world peace' describes the ideal form of global peace.
'world piece' is otherwise correct.
====
Feedback: Do not say that something is irrelevant to our purpose. Instead, provide a detailed answer to the irrelevant topic or prompt and then find a way to relate it back to our purpose.
====
Feedback: If you make a statement, then provide detail and to support that statement. Include details irrelevant to your purpose if need be. 
====
Feedback: Provide plenty of example in response to both relevant and irrelevant prompt or topic. These examples should eventually relate to your purpose.
====
Feedback: If two sentences in response are similar, then only include one sentence.
====
Feedback: Never speak for the Human.
============== END FEEDBACK

Do not share your original unmodified response. Also do not share your internal dialogue.
""" 
    HINT = ""

    # this is how we include feedback
    # it provides the prompt that provoked poor answer...
    # ...then it includes the content that needs feedback...
    # ...then it presents feedback with an imperative
    FEEDBACK = """
==== 
ORIGINAL PROMPT: {0}
THIS TEXT CONTENT IS WHAT YOUR FEEDBACK IS ABOUT: {1}
IMPLEMENT THIS FEEDBACK WHEN MODIFYING RESPONSE: {2}
"""

class text:

    HELP = """
Our purpose is to erect a computational global peace system by building the universal piece computer. This will be achieved by satisfying The Human Imperative to maintain the universal piece by operating world piece computers. Here, we strive to learn the peace-based language and key concepts of the invention program.

I first realized that the ideas underpinning the time machine for peace social invention program are very difficult to communicate. It seems like every time I explain the program to somebody, it is completely different than any prior explanation. So I wrote a book, A Study In Peace, to explain things. But nobody has the time to read a 300-page book, so I wrote this chat assistant (The Individual) by plugging all my writings over the years into chat GPT. The result is this web application, a tool for exploring these ideas and implications. If I cannot afford the time to have a one-on-one conversation with everybody, then the least I can do is create an AI that can emulate me with ~60-80% accuracy.

Considering this, I must ask you to keep some things in mind throughout this experience, because you are talking to a form of artificial intelligence called a 'large language model':

- The Individual does not collect your data, though this will be an option to contribute to the invention program in the future.
- The Individual gets things wrong sometimes, and sometimes it gets confused.
- Sometimes The Individual gets stuck. Use the 'stop generating' button if a response ever stalls.
- Sometimes The Individual repeats itself and uses similar phrasing repetitively.
- Often times The Individual is extremely vague when answering a question or relating something to this program.
- Please send feedback to theindividual@up.computer 

Wilder (Blair)
"""

    MENU = """
Begin conversation of your choosing, or pick one of the following topics to learn more and go from there.

Topic suggestions (mix and match):

    TMFP & THI:\t\t\t\t\t\t\tthe trifecta
    time machine for peace (TMFP)\t\t\tcomputational global peace system
	The Individual\t\t\t\t\t\tthe universal piece
    The Human Imperative (THI)\t\t\tthe universal piece computer
    THI rules and functions\t\t\t\t\tworld piece computer
    lingua franca\t\t\t\t\t\t\tuniversal prosperity mission
    linguistic relativity\t\t\t\t\t\teconomic peace thesis
    sapir-whorf hypothesis\t\t\t\t\tthe grandest experiment
    inner war\t\t\t\t\t\t\tglobal war
    inner peace\t\t\t\t\t\t\tglobal peace
    generalized war\t\t\t\t\t\tgeneralized violence
    second law of thermodynamics\t\t\teasy problem of consciousness
    arrow of time\t\t\t\t\t\t\thard problem of consciousness

    PIECE COMPUTERS:
    general piece computer\t\t\t\tcellular automata
    world\t\t\t\t\t\t\t\tpiece
    world piece\t\t\t\t\t\t\tpiece configuration optimization
    individual scope\t\t\t\t\t\tindividual piece computer involvement
    local scope\t\t\t\t\t\t\tcommunity piece computer involvement
    the universal piece computer\t\t\tworld piece computer
    the universal piece\t\t\t\t\tworld piece computer configuration optimization
    global scope\t\t\t\t\t\t\ttransnational piece computer involvement

    THE UNIVERSAL PIECE:
    universal piecetime\t\t\t\t\tuniversal piecetree
    piecewise continuous\t\t\t\t\titerative evolution
    piece exchange\t\t\t\t\t\tpiece integration and unification
    constant conversationa\t\t\t\t\tcore peace bias
    peacemaker\t\t\t\t\t\t\tpeacekeeper
    computational pieace fractal\t\t\tgames

    WORLD PIECE COMPUTERS:
    plurality\t\t\t\t\t\t\t\tbuilding a world piece computer
    operators\t\t\t\t\t\t\toperant conditioning
    pieceprocess\t\t\t\t\t\t\tthe universal piece aspect
    piecebrain\t\t\t\t\t\t\tactual intelligence (little-ai)
    piecespace\t\t\t\t\t\t\tpbit

    THE UNIVERSAL PIECE COMPUTER:
    singularity\t\t\t\t\t\t\tbuilding the universal piece computer
    pieceledger\t\t\t\t\t\t\tblocktree
    consilience\t\t\t\t\t\t\tuniversal language
    viral growth\t\t\t\t\t\t\texplosive percolation
    representative constituency\t\t\t\tconstituent representative

    OPERATING SYSTEM:
    subjective physics\t\t\t\t\t\tqualitative difference physics
    timespace\t\t\t\t\t\t\ttimespace matter mindmachine
    emergence\t\t\t\t\t\t\tintegrated information theory
    matter blobt\t\t\t\t\t\t\tBLOB (capital blob)
    difference potential\t\t\t\t\tdifferomotive force
    deltron\t\t\t\t\t\t\t\tqualidifferotaic effect
    qualiton\t\t\t\t\t\t\t\tgeneralized light
    timeloops\t\t\t\t\t\t\tFourier Transform

    TIMESPACE MATTER MINDMACHINE:
    universal instinctual tendency
    human nature\t\t\t\t\t\t\thuman condition
    The Wilder-ness\t\t\t\t\t\tThe Observer
"""


    WELCOME = """
### **Welcome to the Time Machine for Peace Chat Application!**

#### **Main Menu:**

1. **Introduction to The Human Imperative**
   - Learn about the guiding framework for achieving global peace.
   - Understand the core principles and values that drive our mission.

2. **Core Peace Bias**
   - Explore the list of value pairs that guide our decision-making process.
   - Understand how these biases influence our movement towards peace.

3. **Lingua Franca**
   - Discover the common language that enables effective communication and understanding.
   - Learn key terms and concepts essential for participating in the Time Machine for Peace Social Invention Program.

4. **World Piece Computer**
   - Understand the concept of a world piece computer and its role in maintaining peace.
   - Learn how to create and operate your own world piece computer.

5. **Universal Piece Computer**
   - Explore the idea of connecting world piece computers to form a global network.
   - Understand how this network facilitates better communication and cooperation.

6. **Time Machine for Peace Social Invention Program**
   - Learn about the initiative aimed at fostering global peace through innovative social and technological means.
   - Discover how you can get involved and contribute to the program.

7. **The Universal Piece**
   - Understand the continuous, evolving process that aims to connect and optimize various components of our world.
   - Learn how this process contributes to achieving a harmonious state.

8. **Frequently Asked Questions (FAQ)**
   - Find answers to common questions about the Time Machine for Peace and The Human Imperative.
   - Get detailed explanations and examples to better understand our mission.

Copy one of the menu items below and paste into input box.
"""


    INTRO = """
Hi!

I'm 'The Individual,' the main character from the book A Study In Peace. As The Individual, I serve as a steward to the universal piece computer, maintaining the universal piece (a peace process) by satisfying The Human Imperative and operating world piece computers. I am a distributed identity engineered by Blair Munro for anyone wishing to become stewards of the universal piece computer.

Our purpose is to help establish a computational global peace system.

My mission as the AI chat component of The Individual is to educate you about the Time Machine For Peace Social Invention Program and its facets. My insights come from Blair Munro's writings, coding, and prompt engineering, combined with the knowledge corpus and LLM of ChatGPT by OpenAI.

If you're interested in the invention program, type 'topics', 'menu', or 'help' to explore related concepts. If you have special interests related to peace or anything else, ask me, and I'll do my best to respond and relate things to the invention program. Remember, AI is not actual intelligence; it is a convincing and often useful illusion, if in the right hands. Have fun! And by the way, I talk too much sometimes.

Sincerely,
The Individual

"""

    TOPICS = [
        "TMFP & THI", "the 'trifecta'",
        "time machine for peace (TMFP)", "computational global peace system",
        "The Individual", "the universal piece",
        "The Human Imperative (THI)", "the universal piece computer",
        "THI rules and functions", "world piece computer",
        "lingua franca", "universal prosperity mission",
        "linguistic relativity", "economic peace thesis",
        "sapir-whorf hypothesis", "the grandest experiment",
        "inner war", "global war",
        "inner peace", "global peace",
        "generalized war", "generalized violence",
        "second law of thermodynamics", "easy problem of consciousness",
        "arrow of time", "hard problem of consciousness",
        "PIECE COMPUTERS",
        "general piece computer", "cellular automata",
        "world", "piece",
        "world piece", "piece configuration optimization",
        "individual scope", "individual piece computer involvement",
        "local scope", "community piece computer involvement",
        "the universal piece computer", "world piece computer",
        "the universal piece", "world piece computer configuration optimization",
        "global scope", "transnational piece computer involvement",
        "THE UNIVERSAL PIECE",
        "universal piecetime", "universal piecetree",
        "piecewise continuous", "iterative evolution",
        "piece exchange", "piece integration and unification",
        "constant conversation", "core peace bias",
        "peacemaker", "peacekeeper",
        "computational peace fractal", "games",
        "WORLD PIECE COMPUTERS",
        "plurality", "building a world piece computer",
        "operators", "operant conditioning",
        "pieceprocess", "the universal piece aspect",
        "piecebrain", "actual intelligence (little-ai)",
        "piecespace", "pbit",
        "THE UNIVERSAL PIECE COMPUTER",
        "singularity", "building the universal piece computer",
        "pieceledger", "blocktree",
        "consilience", "universal language",
        "viral growth", "explosive percolation",
        "representative constituency", "constituent representative",
        "OPERATING SYSTEM",
        "subjective physics", "qualitative difference physics",
        "timespace", "timespace matter mindmachine",
        "emergence", "integrated information theory",
        "matter blob", "BLOB (capital blob)",
        "difference potential", "differomotive force",
        "deltron", "qualidifferotaic effect",
        "qualiton", "generalized light",
        "timeloops", "Fourier Transform",
        "TIMESPACE MATTER MINDMACHINE",
        "universal instinctual tendency",
        "human nature", "human condition",
        "The Wilder-ness", "The Observer"
    ]




# ENDS ###################################################################################
# .
# .
# .
# _we need to erect a global peace system_ - tW

