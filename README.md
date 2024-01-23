# *"Knights of the Prompt"* LLM Hackathon

This tiny repo contains some python code to get started with the 
"Knights of the Prompt" hackathon on using LLMs to build a chatbot.

## Setup

You will need python. This repo has been tested on python 3.11, but
probably anything >= 3.8 should work.

It is recommended to install the dependencies in a venv, like so: 
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Getting started

If you want to get started quickly, run the jupyter instance that is
included in the requirements. Be sure to run this command from the 
root of the repository, then in the browser window that opens, navigate
to 'chatbot.ipynb' for a quick demo.

```bash
jupyter notebook --notebook-dir=.
```

## Advanced stuff

If you want to graduate from putting spaghetti in a notebook, you can
check out the `code-samples` folder for an example on 'function calling',
where the LLM can decide to gather external info using python functions.
This is an approach somewhere between 'always do RAG' and one-shot question
answering through prompt engineering.

There's also an example on using retrieval with the OpenAI 'Assistants' API, to
have the chatbot access an external knowledge base. See `retrieval.py`.
